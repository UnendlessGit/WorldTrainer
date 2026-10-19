import AVFoundation
import Foundation
import MLX
import MLXNN

/// Headless verification of the whole learning pipeline.
///
/// Launch with `--self-test` (optionally `--self-test-package <path>`) to run
/// import, dataset preparation, training with pause and resume, checkpoint
/// verification, generation and a per-resolution speed benchmark, then print a
/// JSON report. Nothing here is simulated: it exercises the same code paths the
/// interface uses.
enum SelfTest {
    struct Report: Codable {
        var startedAt = Date()
        var packagePath = ""
        var importIssues: [String] = []
        var datasetFrames = 0
        var validationFrames = 0
        var parameterCount = 0
        var weightsChanged = false
        var firstLoss: Float?
        var lastLoss: Float?
        var validationLoss: Float?
        var checkpointKinds: [String] = []
        var pausedCheckpointStep: Int?
        var cancelledCheckpointStep: Int?
        var resumedFromStep: Int?
        var resumedPhase = ""
        var completedPhase = ""
        var checks: [String: Bool] = [:]
        var generationWarmupFrames = 0
        var generatedFrames = 0
        var generationMillisecondsPerFrame = 0.0
        var generationFPS = 0.0
        var frameVariation = 0.0
        var capabilities: [String] = []
        var benchmark: [Benchmark] = []
        var log: [String] = []
        var failure: String?
    }

    struct Benchmark: Codable {
        var resolution: String
        var millisecondsPerFrame: Double
        var framesPerSecond: Double
        var parameterCount: Int
    }

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var value = Report()
        func mutate(_ body: (inout Report) -> Void) {
            lock.lock()
            body(&value)
            lock.unlock()
        }
        func read() -> Report {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    static func runAndExit() -> Never {
        let arguments = CommandLine.arguments
        let packagePath: String
        if let index = arguments.firstIndex(of: "--self-test-package"), index + 1 < arguments.count {
            packagePath = arguments[index + 1]
        } else {
            packagePath = FileManager.default.currentDirectoryPath + "/RecordingFormat"
        }

        let box = Box()
        box.mutate {
            $0.packagePath = packagePath
            $0.log.append("WorldTrainer self test")
        }

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("WorldTrainerSelfTest-\(UUID().uuidString)", isDirectory: true)
        let directories = AppDirectories(root: root)

        do {
            try run(packagePath: packagePath, directories: directories, box: box)
        } catch {
            box.mutate {
                $0.failure = error.localizedDescription
                $0.log.append("FAILED: \(error.localizedDescription)")
            }
        }

        let report = box.read()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        var final = report
        final.failure = report.failure ?? (report.checks.values.contains(false)
            ? "One or more verification checks failed: \(report.checks.filter { !$0.value }.keys.sorted().joined(separator: ", "))"
            : nil)
        if let data = try? encoder.encode(final), let text = String(data: data, encoding: .utf8) {
            print(text)
            try? data.write(to: root.appendingPathComponent("self-test-report.json"))
        }
        print(final.failure == nil ? "SELF TEST PASSED" : "SELF TEST FAILED")
        exit(final.failure == nil ? 0 : 1)
    }

    // MARK: - Pipeline

    private static func run(packagePath: String, directories: AppDirectories, box: Box) throws {
        try directories.prepare()
        let packageURL = URL(fileURLWithPath: packagePath)
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw WorldTrainerError.invalidData("No recording package at \(packagePath)")
        }

        // 1. Inspect and import, exactly like the Library tab does.
        let inspection = try awaitValue { await RecordingInspector.inspect(packageDirectory: packageURL) }
        guard let manifest = inspection.manifest, inspection.isUsable else {
            throw WorldTrainerError.invalidData("The package is not usable: \(inspection.issues.map(\.message).joined(separator: " "))")
        }
        box.mutate {
            $0.importIssues = inspection.issues.map { "\($0.severity.rawValue): \($0.message)" }
            $0.capabilities = inspection.eventSummary?.capabilities.summaryLines ?? []
            $0.log.append("Imported \(manifest.name) · \(manifest.duration.formatted(.number.precision(.fractionLength(2))))s · \(manifest.eventCount) events")
        }

        let recordingID = UUID()
        let destination = directories.recordingsDirectory.appendingPathComponent(recordingID.uuidString, isDirectory: true)
        try FileManager.default.copyItem(at: packageURL, to: destination)
        let source = DatasetRecordingSource(
            recordingID: recordingID,
            name: manifest.name,
            directory: destination,
            manifest: manifest,
            videoDuration: inspection.videoDuration)

        // 2. Configuration used for the verification run.
        var configuration = WorldModelConfiguration()
        configuration.resolution = .compact
        configuration.sequenceLength = 8
        configuration.baseChannels = 24
        configuration.recurrentChannels = 64
        configuration.actionEmbeddingChannels = 16
        configuration.batchSize = 2
        configuration.epochs = 2
        configuration.learningRate = 0.0006
        configuration.validationFraction = 0.1

        let model = WorldModelRecord(
            id: UUID(),
            name: "Self test model",
            createdAt: Date(),
            updatedAt: Date(),
            configuration: configuration,
            recordingIDs: [recordingID])

        // 3. Dataset preparation.
        let fingerprint = DatasetBuilder.fingerprint(configuration: configuration, sources: [source])
        let cacheDirectory = directories.trainingCache(modelID: model.id, fingerprint: fingerprint)
        let datasetManifest = try DatasetBuilder.build(
            sources: [source],
            configuration: configuration,
            fingerprint: fingerprint,
            directory: cacheDirectory,
            progress: { _, _ in },
            checkCancellation: {})
        box.mutate {
            $0.datasetFrames = datasetManifest.trainingFrameCount
            $0.validationFrames = datasetManifest.validationFrameCount
            $0.parameterCount = configuration.estimatedParameterCount()
            $0.log.append("Dataset ready · \(datasetManifest.trainingFrameCount) training frames, \(datasetManifest.validationFrameCount) held out")
        }

        // 4. Training with a deliberate pause, then resume, then a stop.
        let initialChecksum = try weightChecksum(modelID: model.id, configuration: configuration, checkpoint: nil)

        let first = try train(
            model: model,
            configuration: configuration,
            sources: [source],
            directories: directories,
            resume: false,
            pauseAfterStep: 40,
            cancelAfterStep: nil,
            box: box)
        box.mutate {
            $0.firstLoss = first.firstLoss
            $0.lastLoss = first.lastLoss
            $0.pausedCheckpointStep = first.checkpointStep
            $0.log.append("Paused at step \(first.steps) with checkpoint step \(first.checkpointStep.map(String.init) ?? "none")")
        }

        let second = try train(
            model: model,
            configuration: configuration,
            sources: [source],
            directories: directories,
            resume: true,
            pauseAfterStep: nil,
            cancelAfterStep: 25,
            box: box)
        box.mutate {
            $0.cancelledCheckpointStep = second.checkpointStep
            $0.resumedFromStep = second.resumedFromStep
            $0.resumedPhase = second.phase
            $0.log.append("Resumed from step \(second.resumedFromStep ?? -1) and stopped at checkpoint step \(second.checkpointStep.map(String.init) ?? "none")")
            $0.checks["resume-continues-from-paused-checkpoint"] = second.resumedFromStep == first.checkpointStep
            $0.checks["stop-writes-a-checkpoint"] = (second.checkpointStep ?? -1) > (first.checkpointStep ?? -1)
            $0.checks["pause-writes-a-checkpoint"] = first.checkpointStep != nil
        }

        let third = try train(
            model: model,
            configuration: configuration,
            sources: [source],
            directories: directories,
            resume: true,
            pauseAfterStep: nil,
            cancelAfterStep: nil,
            box: box)
        let checkpoints = CheckpointStore.list(modelID: model.id, directories: directories)
        box.mutate {
            $0.validationLoss = third.validationLoss
            $0.completedPhase = third.phase
            $0.checkpointKinds = checkpoints.map { "\($0.kind.rawValue)@\($0.step)" }
            $0.log.append("Training finished in phase \(third.phase) with \(checkpoints.count) checkpoints")
            $0.checks["training-completes"] = third.phase == "Complete"
            $0.checks["validation-loss-is-finite"] = (third.validationLoss ?? .infinity).isFinite
            $0.checks["validation-uses-held-out-frames"] = $0.validationFrames > 0
        }

        // 5. Weights must have moved away from the random initialisation.
        guard let latest = CheckpointStore.latest(
            modelID: model.id,
            configurationFingerprint: configuration.architectureFingerprint,
            datasetFingerprint: fingerprint,
            directories: directories) else {
            throw WorldTrainerError.training("No checkpoint was produced by the training run.")
        }
        let trainedChecksum = try weightChecksum(modelID: model.id, configuration: configuration, checkpoint: latest)
        box.mutate {
            $0.weightsChanged = trainedChecksum != initialChecksum
            $0.checks["weights-changed-during-training"] = trainedChecksum != initialChecksum
            $0.checks["loss-decreased"] = (third.lastLoss ?? 1) < (first.firstLoss ?? 0)
            $0.log.append("Weight checksum \(initialChecksum.prefix(12)) → \(trainedChecksum.prefix(12))")
        }

        // 6. Generation from the trained checkpoint.
        let capabilities = inspection.eventSummary?.capabilities ?? ActionCapabilities()
        let seed = try awaitValue {
            try await GenerationSeedBuilder.build(
                directory: destination,
                manifest: manifest,
                configuration: configuration,
                warmupFrames: 24,
                capabilities: capabilities)
        }
        let engine = try GenerationEngine(
            checkpoint: latest.checkpoint,
            checkpointDirectory: latest.directory,
            seed: seed,
            capabilities: capabilities)

        var actionRow = [Float](repeating: 0, count: ActionLayout.count)
        var firstFrame: [UInt8] = []
        var lastFrame: [UInt8] = []
        let started = CFAbsoluteTimeGetCurrent()
        let frames = 60
        for index in 0..<frames {
            // Drive the model with a plausible held control so the rollout is
            // not just a static scene.
            actionRow[ActionLayout.keyboard.lowerBound] = index % 20 < 10 ? 1 : 0
            actionRow[ActionLayout.relativeMouse.lowerBound] = Float(index % 7) / 7
            let bytes = engine.step(action: actionRow)
            if index == 0 { firstFrame = bytes }
            if index == frames - 1 { lastFrame = bytes }
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - started
        let variation = zip(firstFrame, lastFrame).map { abs(Double($0) - Double($1)) }.reduce(0, +) / Double(max(1, firstFrame.count))
        box.mutate {
            $0.generationWarmupFrames = engine.warmupFrameCount
            $0.generatedFrames = engine.generatedFrameCount
            $0.generationMillisecondsPerFrame = elapsed / Double(frames) * 1000
            $0.generationFPS = Double(frames) / elapsed
            $0.frameVariation = variation
            $0.checks["generation-produces-changing-frames"] = variation > 0.5
            $0.checks["generation-runs-faster-than-15fps"] = Double(frames) / elapsed >= 15
            $0.checks["generation-warms-up-from-recording"] = engine.warmupFrameCount > 0
            $0.log.append("Generated \(frames) frames at \(String(format: "%.1f", Double(frames) / elapsed)) FPS (\(String(format: "%.1f", elapsed / Double(frames) * 1000)) ms/frame)")
        }

        // 7. Speed benchmark across the resolutions the interface offers.
        for resolution in ModelResolution.allCases {
            var benchConfiguration = WorldModelConfiguration()
            benchConfiguration.resolution = resolution
            let benchmark = try benchmarkModel(configuration: benchConfiguration, steps: 40)
            box.mutate { $0.benchmark.append(benchmark) }
        }
        box.mutate {
            $0.checks["every-resolution-exceeds-15fps"] = $0.benchmark.allSatisfy { $0.framesPerSecond >= 15 }
        }
    }

    private struct TrainingOutcome {
        var steps = 0
        var phase = ""
        var firstLoss: Float?
        var lastLoss: Float?
        var validationLoss: Float?
        var checkpointStep: Int?
        var resumedFromStep: Int?
    }

    private static func train(
        model: WorldModelRecord,
        configuration: WorldModelConfiguration,
        sources: [DatasetRecordingSource],
        directories: AppDirectories,
        resume: Bool,
        pauseAfterStep: Int?,
        cancelAfterStep: Int?,
        box: Box
    ) throws -> TrainingOutcome {
        let control = TrainingControl()
        let lock = NSLock()
        var outcome = TrainingOutcome()
        var observedSteps = 0
        var resumedFrom: Int?

        let publish: @Sendable (TrainingProgress) -> Void = { progress in
            lock.lock()
            outcome.phase = progress.phase.rawValue
            outcome.lastLoss = progress.trainingLoss
            outcome.validationLoss = progress.validationLoss
            if outcome.firstLoss == nil, progress.trainingLoss != nil { outcome.firstLoss = progress.trainingLoss }
            if progress.message.hasPrefix("Resumed from") { resumedFrom = progress.step }
            lock.unlock()

            if progress.phase == .training {
                observedSteps += 1
                if let pauseAfterStep, observedSteps == pauseAfterStep { control.set(.pause) }
                if let cancelAfterStep, observedSteps == cancelAfterStep { control.set(.cancel) }
            }
        }

        var checkpointSteps: [Int] = []
        let saved: @Sendable (TrainingCheckpoint) -> Void = { checkpoint in
            lock.lock()
            checkpointSteps.append(checkpoint.step)
            lock.unlock()
        }

        let request = TrainingRequest(
            model: model,
            configuration: configuration,
            sources: sources,
            directories: directories,
            resume: resume,
            memoryLimitGB: 24,
            cacheLimitGB: 4)

        let semaphore = DispatchSemaphore(value: 0)
        TrainingWorker.queue.async {
            TrainingWorker.run(request, control: control, publish: publish, checkpointSaved: saved)
            semaphore.signal()
        }
        semaphore.wait()

        lock.lock()
        outcome.steps = observedSteps
        outcome.checkpointStep = checkpointSteps.max()
        outcome.resumedFromStep = resumedFrom
        if outcome.phase == "Failed" { outcome.firstLoss = nil }
        let result = outcome
        lock.unlock()

        if result.phase == "Failed" {
            throw WorldTrainerError.training("The training worker reported a failure during the self test.")
        }
        return result
    }

    private static func benchmarkModel(configuration: WorldModelConfiguration, steps: Int) throws -> Benchmark {
        Memory.memoryLimit = 24 * 1_073_741_824
        let model = WorldModel(configuration: configuration)
        model.train(false)
        let width = configuration.width
        let height = configuration.height
        var image = MLXRandom.normal([1, height, width, 3])
        var hidden = model.emptyHidden(batchSize: 1)
        var action = MLXRandom.normal([1, ActionLayout.count])
        eval(image, hidden)
        // Warm up Metal kernels before timing.
        for _ in 0..<5 {
            let (next, state) = model.step(image: image, action: action, hidden: hidden)
            eval(next)
            image = next
            hidden = state
        }
        let started = CFAbsoluteTimeGetCurrent()
        for _ in 0..<steps {
            let (next, state) = model.step(image: image, action: action, hidden: hidden)
            eval(next)
            image = next
            hidden = state
            action = MLXRandom.normal([1, ActionLayout.count])
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - started
        let perFrame = elapsed / Double(steps)
        return Benchmark(
            resolution: configuration.resolution.rawValue,
            millisecondsPerFrame: perFrame * 1000,
            framesPerSecond: 1 / perFrame,
            parameterCount: configuration.estimatedParameterCount())
    }

    /// Loads a checkpoint's weights and hashes them, or hashes a freshly
    /// initialised model when no checkpoint is given.
    private static func weightChecksum(
        modelID: UUID,
        configuration: WorldModelConfiguration,
        checkpoint: (checkpoint: TrainingCheckpoint, directory: URL)?
    ) throws -> String {
        let model = WorldModel(configuration: configuration)
        if let checkpoint {
            try CheckpointStore.restoreWeights(model, from: checkpoint.directory)
        }
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for (key, value) in model.parameters().flattened().sorted(by: { $0.0 < $1.0 }) {
            let values = value.asArray(Float.self)
            hash = (hash ^ UInt64(truncatingIfNeeded: values.count)) &* 0x100_0000_01b3
            for sample in values where sample.isFinite {
                hash = (hash ^ UInt64(truncatingIfNeeded: Int64(sample * 1000))) &* 0x100_0000_01b3
            }
            _ = key
        }
        return String(hash, radix: 16)
    }

    private static func awaitValue<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<T, Error>?
        Task.detached(priority: .userInitiated) {
            do {
                result = .success(try await body())
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        switch result {
        case .success(let value): return value
        case .failure(let error): throw error
        case nil: throw WorldTrainerError.invalidData("The background task produced no result.")
        }
    }
}
