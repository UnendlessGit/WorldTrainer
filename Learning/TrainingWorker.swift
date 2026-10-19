import Foundation
import MLX
import MLXNN
import MLXRandom

enum TrainingPhase: String, Sendable {
    case idle = "Ready"
    case preparing = "Preparing data"
    case training = "Training"
    case validating = "Validating"
    case checkpointing = "Saving checkpoint"
    case paused = "Paused"
    case cancelled = "Cancelled"
    case complete = "Complete"
    case failed = "Failed"

    var isBusy: Bool {
        [.preparing, .training, .validating, .checkpointing].contains(self)
    }
}

struct LossPoint: Identifiable, Sendable {
    var step: Int
    var training: Float
    var validation: Float?
    var id: Int { step }
}

struct TrainingProgress: Sendable {
    var phase: TrainingPhase = .idle
    var message = "Select a model and start training."
    var modelID: UUID?
    var epoch = 0
    var epochs = 0
    var step = 0
    var cursor = 0
    var stepsPerEpoch = 0
    var samplesPerEpoch = 0
    var trainingLoss: Float?
    var validationLoss: Float?
    var bestValidationLoss: Float?
    var learningRate: Float = 0
    var gradientNorm: Float?
    var stepsPerSecond = 0.0
    var elapsedSeconds = 0.0
    var activeMemory = 0
    var cacheMemory = 0
    var datasetFrames = 0
    var validationFrames = 0
    var checkpointID: UUID?
    var checkpointKind: ModelCheckpointKind?
    var history: [LossPoint] = []
    var error: String?

    /// Step index inside the current epoch, derived from the global step count.
    var stepWithinEpoch: Int {
        guard stepsPerEpoch > 0 else { return 0 }
        return max(0, step - epoch * stepsPerEpoch)
    }
}

final class TrainingControl: @unchecked Sendable {
    enum Request { case run, pause, cancel }

    private let lock = NSLock()
    private var request: Request = .run

    func set(_ value: Request) {
        lock.lock()
        request = value
        lock.unlock()
    }

    var current: Request {
        lock.lock()
        defer { lock.unlock() }
        return request
    }
}

struct TrainingRequest: Sendable {
    let model: WorldModelRecord
    let configuration: WorldModelConfiguration
    let sources: [DatasetRecordingSource]
    let directories: AppDirectories
    let resume: Bool
    let memoryLimitGB: Double
    let cacheLimitGB: Double
}

/// All MLX arrays and modules are confined to one serial worker. The main actor
/// receives value snapshots only.
enum TrainingWorker {
    static let queue = DispatchQueue(label: "com.worldtrainer.training", qos: .userInitiated)

    static func run(
        _ request: TrainingRequest,
        control: TrainingControl,
        publish: @escaping @Sendable (TrainingProgress) -> Void,
        checkpointSaved: @escaping @Sendable (TrainingCheckpoint) -> Void
    ) {
        let started = Date()
        var progress = TrainingProgress(
            phase: .preparing,
            message: "Checking the selected recordings…",
            modelID: request.model.id,
            epochs: request.configuration.epochs,
            learningRate: Float(request.configuration.learningRate))
        publish(progress)

        func checkCancellation() throws {
            if control.current == .cancel { throw CancellationError() }
        }

        do {
            let issues = request.configuration.validationIssues()
            guard issues.isEmpty else {
                throw WorldTrainerError.invalidData(issues.joined(separator: " "))
            }
            let usable = request.sources.filter { $0.manifest.isStructurallyValid }
            guard !usable.isEmpty else {
                throw WorldTrainerError.invalidData("None of the selected recordings are usable for training.")
            }
            Memory.memoryLimit = Int(max(2, request.memoryLimitGB) * 1_073_741_824)
            Memory.cacheLimit = Int(max(0.25, request.cacheLimitGB) * 1_073_741_824)
            try request.directories.prepare()

            let fingerprint = DatasetBuilder.fingerprint(configuration: request.configuration, sources: usable)
            let cacheDirectory = request.directories.trainingCache(modelID: request.model.id, fingerprint: fingerprint)
            let datasetManifest: DatasetCacheManifest
            if let existing = try? loadDatasetManifest(at: cacheDirectory), existing.fingerprint == fingerprint,
               FileManager.default.fileExists(atPath: cacheDirectory.appendingPathComponent("frames.bin").path),
               FileManager.default.fileExists(atPath: cacheDirectory.appendingPathComponent("actions.bin").path) {
                datasetManifest = existing
                progress.message = "Reusing the prepared dataset cache."
                progress.datasetFrames = existing.trainingFrameCount
                progress.validationFrames = existing.validationFrameCount
                publish(progress)
            } else {
                progress.phase = .preparing
                progress.message = "Preparing dataset…"
                publish(progress)
                datasetManifest = try DatasetBuilder.build(
                    sources: usable,
                    configuration: request.configuration,
                    fingerprint: fingerprint,
                    directory: cacheDirectory,
                    progress: { fraction, message in
                        progress.phase = .preparing
                        progress.message = message
                        progress.datasetFrames = Int(Double(datasetManifestEstimate(usable, configuration: request.configuration)) * fraction)
                        publish(progress)
                    },
                    checkCancellation: checkCancellation)
            }
            try checkCancellation()

            progress.datasetFrames = datasetManifest.trainingFrameCount
            progress.validationFrames = datasetManifest.validationFrameCount
            progress.message = datasetManifest.validationFrameCount > 0
                ? "Dataset ready. Validation uses \(datasetManifest.validationFrameCount) held-out frames."
                : "Dataset ready. Select two or more recordings for held-out validation."
            publish(progress)

            let reader = try DatasetReader(directory: cacheDirectory, manifest: datasetManifest)
            MLXRandom.seed(request.configuration.seed)
            let model = WorldModel(configuration: request.configuration)
            let optimizer = ResumableAdamW(
                learningRate: Float(request.configuration.learningRate),
                weightDecay: Float(request.configuration.weightDecay))

            var epoch = 0
            var cursor = 0
            var step = 0
            var bestValidationLoss: Float?
            var history: [LossPoint] = []
            var initialCheckpointID: UUID?

            if request.resume,
               let latest = CheckpointStore.latest(
                   modelID: request.model.id,
                   configurationFingerprint: request.configuration.architectureFingerprint,
                   datasetFingerprint: fingerprint,
                   directories: request.directories) {
                try CheckpointStore.restoreWeights(model, from: latest.directory)
                try CheckpointStore.restoreOptimizer(optimizer, model: model, from: latest.directory)
                epoch = latest.checkpoint.epoch
                cursor = latest.checkpoint.sampleCursor
                step = latest.checkpoint.step
                bestValidationLoss = latest.checkpoint.bestValidationLoss ?? latest.checkpoint.validationLoss
                initialCheckpointID = latest.checkpoint.id
                progress.checkpointID = latest.checkpoint.id
                progress.checkpointKind = latest.checkpoint.kind
                progress.epoch = epoch
                progress.step = step
                progress.cursor = cursor
                progress.samplesPerEpoch = latest.checkpoint.epochSampleCount
                progress.bestValidationLoss = bestValidationLoss
                progress.message = "Resumed from \(latest.checkpoint.label) at step \(step)."
                publish(progress)
                if epoch >= request.configuration.epochs {
                    throw WorldTrainerError.invalidData("This run already completed all configured epochs. Increase the epoch count to keep training.")
                }
            }

            let lossGradient = valueAndGrad(model: model) { model, frames, actions in
                WorldModelLoss.loss(
                    model: model,
                    frames: frames,
                    actions: actions,
                    configuration: request.configuration)
            }

            var lastPublish = Date.distantPast
            var lastLoss: Float = 0
            var lastGradientNorm: Float?
            var samplesPerEpoch = 0

            func report(_ phase: TrainingPhase, _ message: String, force: Bool = false) {
                progress.phase = phase
                progress.message = message
                progress.epoch = epoch
                progress.step = step
                progress.cursor = cursor
                progress.trainingLoss = step > 0 ? lastLoss : nil
                progress.bestValidationLoss = bestValidationLoss
                progress.gradientNorm = lastGradientNorm
                progress.activeMemory = Memory.activeMemory
                progress.cacheMemory = Memory.cacheMemory
                progress.elapsedSeconds = Date().timeIntervalSince(started)
                progress.stepsPerSecond = Double(step) / max(0.001, progress.elapsedSeconds)
                progress.history = history
                let now = Date()
                if force || now.timeIntervalSince(lastPublish) > 0.1 {
                    lastPublish = now
                    publish(progress)
                }
            }

            func saveCheckpoint(kind: ModelCheckpointKind, validationLoss: Float?) throws {
                let checkpoint = TrainingCheckpoint(
                    id: UUID(),
                    modelID: request.model.id,
                    createdAt: Date(),
                    step: step,
                    epoch: epoch,
                    sampleCursor: cursor,
                    epochSampleCount: samplesPerEpoch,
                    trainingLoss: step > 0 ? lastLoss : nil,
                    validationLoss: validationLoss,
                    bestValidationLoss: bestValidationLoss,
                    configurationFingerprint: request.configuration.architectureFingerprint,
                    datasetFingerprint: fingerprint,
                    datasetRecordingIDs: datasetManifest.recordingIDs,
                    trainingRecordingIDs: datasetManifest.trainingRecordingIDs,
                    validationRecordingIDs: datasetManifest.validationRecordingIDs,
                    configuration: request.configuration,
                    seed: request.configuration.seed,
                    kind: kind,
                    parameterCount: request.configuration.estimatedParameterCount())
                report(.checkpointing, "Writing weights and optimizer state…", force: true)
                let saved = try CheckpointStore.save(
                    checkpoint: checkpoint, model: model, optimizer: optimizer,
                    directories: request.directories)
                progress.checkpointID = saved.id
                progress.checkpointKind = saved.kind
                checkpointSaved(saved)
            }

            /// Stops the run and writes a resumable checkpoint first, so a stop
            /// never loses work. Returns true when the run was stopped.
            func stopWithCheckpoint() throws -> Bool {
                guard control.current == .cancel else { return false }
                try saveCheckpoint(kind: .cancelled, validationLoss: progress.validationLoss)
                report(.cancelled, "Stopped safely. The newest checkpoint keeps everything learned so far.", force: true)
                return true
            }

            while epoch < request.configuration.epochs {
                if try stopWithCheckpoint() { return }
                let schedule = TrainingSchedule(
                    manifest: datasetManifest,
                    sequenceLength: request.configuration.sequenceLength,
                    batchSize: request.configuration.batchSize,
                    epoch: epoch,
                    seed: request.configuration.seed)
                samplesPerEpoch = schedule.samples.count
                progress.samplesPerEpoch = samplesPerEpoch
                progress.stepsPerEpoch = schedule.stepCount
                let startingStep = min(schedule.stepCount, cursor / max(1, request.configuration.batchSize))
                var stepIndex = startingStep

                if samplesPerEpoch == 0 {
                    throw WorldTrainerError.invalidData("The prepared dataset contains no trainable sequences.")
                }

                while stepIndex < schedule.stepCount {
                    if control.current == .pause {
                        try saveCheckpoint(kind: .paused, validationLoss: progress.validationLoss)
                        report(.paused, "Paused with a resumable checkpoint.", force: true)
                        return
                    }
                    if try stopWithCheckpoint() { return }
                    let batchSamples = Array(schedule.batch(atStep: stepIndex))
                    guard !batchSamples.isEmpty else { break }
                    let batch = reader.batch(samples: batchSamples, sequenceLength: request.configuration.sequenceLength)
                    let (lossArray, gradients) = lossGradient(model, batch.frames, batch.actions)
                    let loss = lossArray.item(Float.self)
                    guard loss.isFinite else {
                        throw WorldTrainerError.training("Training loss became non-finite at step \(step + 1). The previous checkpoint is preserved.")
                    }
                    lastGradientNorm = optimizer.update(
                        model: model, gradients: gradients,
                        clip: Float(request.configuration.gradientClip)).item(Float.self)
                    step += 1
                    lastLoss = loss
                    cursor = min(samplesPerEpoch, (stepIndex + 1) * request.configuration.batchSize)
                    history.append(LossPoint(step: step, training: loss, validation: history.last?.validation))
                    if history.count > 400 { history.removeFirst(history.count - 400) }
                    report(.training, "Epoch \(epoch + 1)/\(request.configuration.epochs) · step \(stepIndex + 1)/\(schedule.stepCount)")
                    if step % 200 == 0 {
                        try saveCheckpoint(kind: .periodic, validationLoss: progress.validationLoss)
                    }
                    stepIndex += 1
                }

                // Epoch boundary: evaluate held-out frames when available.
                if try stopWithCheckpoint() { return }
                let validationSamples = schedule.validationSamples(
                    manifest: datasetManifest,
                    sequenceLength: request.configuration.sequenceLength)
                var validationLoss: Float?
                if !validationSamples.isEmpty {
                    model.train(false)
                    var weighted = 0.0
                    var count = 0
                    var index = 0
                    while index < validationSamples.count {
                        if try stopWithCheckpoint() { return }
                        let slice = Array(validationSamples[index..<min(validationSamples.count, index + request.configuration.batchSize)])
                        let batch = reader.batch(samples: slice, sequenceLength: request.configuration.sequenceLength)
                        let value = WorldModelLoss.loss(
                            model: model, frames: batch.frames, actions: batch.actions,
                            configuration: request.configuration).item(Float.self)
                        guard value.isFinite else {
                            throw WorldTrainerError.training("Validation loss became non-finite. The previous checkpoint is preserved.")
                        }
                        weighted += Double(value) * Double(slice.count)
                        count += slice.count
                        index += request.configuration.batchSize
                        progress.phase = .validating
                        progress.message = "Validating \(index)/\(validationSamples.count) held-out windows…"
                        publish(progress)
                    }
                    model.train(true)
                    validationLoss = count > 0 ? Float(weighted / Double(count)) : nil
                }

                epoch += 1
                cursor = 0
                progress.validationLoss = validationLoss
                if let validationLoss, bestValidationLoss == nil || validationLoss < bestValidationLoss! {
                    bestValidationLoss = validationLoss
                    history.append(LossPoint(step: step, training: lastLoss, validation: validationLoss))
                    try saveCheckpoint(kind: .best, validationLoss: validationLoss)
                } else {
                    history.append(LossPoint(step: step, training: lastLoss, validation: validationLoss))
                    try saveCheckpoint(kind: .epoch, validationLoss: validationLoss)
                }
                report(.training, "Completed epoch \(epoch)/\(request.configuration.epochs).", force: true)
            }

            try saveCheckpoint(kind: .epoch, validationLoss: progress.validationLoss)
            report(.complete, "Training complete. \(step) steps · final loss \(lastLoss.formatted(.number.precision(.fractionLength(4)))).", force: true)
        } catch is CancellationError {
            progress.phase = .cancelled
            progress.message = "Stopped before training began. Nothing was lost."
            progress.error = nil
            publish(progress)
        } catch {
            progress.phase = .failed
            progress.error = error.localizedDescription
            progress.message = error.localizedDescription
            publish(progress)
        }
    }

    private static func loadDatasetManifest(at directory: URL) throws -> DatasetCacheManifest {
        let data = try Data(contentsOf: directory.appendingPathComponent("dataset.json"))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(DatasetCacheManifest.self, from: data)
    }

    private static func datasetManifestEstimate(_ sources: [DatasetRecordingSource], configuration: WorldModelConfiguration) -> Int {
        sources.reduce(0) { total, source in
            let start = source.manifest.effectiveTrimStart
            let end = max(start, source.manifest.effectiveTrimEnd)
            return total + max(0, Int((end - start) * configuration.targetFPS))
        }
    }
}
