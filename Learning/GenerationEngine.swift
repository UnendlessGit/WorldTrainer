import AVFoundation
import CoreGraphics
import Foundation
import MLX
import MLXNN

/// Warm-up context for a Run session.
///
/// The frames here are real footage and are used only to fill the model's
/// persistent memory before generation begins. Every frame shown after warm-up
/// is produced by the model.
struct GenerationSeed: Sendable {
    var frames: [[UInt8]]
    var actions: [[Float]]
    var recordingName: String
    var startSeconds: Double
    var capabilities: ActionCapabilities

    var warmupFrameCount: Int { max(0, frames.count - 1) }
}

enum GenerationSeedBuilder {
    /// Decodes the first frames of a recording at the model resolution together
    /// with the control rows that were recorded alongside them.
    static func build(
        directory: URL,
        manifest: RecordingManifest,
        configuration: WorldModelConfiguration,
        warmupFrames: Int,
        startSeconds: Double? = nil,
        capabilities: ActionCapabilities
    ) async throws -> GenerationSeed {
        let frames = max(2, warmupFrames + 1)
        let name = manifest.name
        let width = configuration.width
        let height = configuration.height
        let fps = configuration.targetFPS
        let start = startSeconds ?? manifest.effectiveTrimStart

        return try await Task.detached(priority: .userInitiated) {
            let videoURL = directory.appendingPathComponent(manifest.videoFile)
            let info = try await VideoFrames.loadTrack(url: videoURL)
            let available = max(0, manifest.effectiveTrimEnd - start)
            let wanted = Double(frames) / fps
            let end = start + min(available, wanted)
            guard end > start else {
                throw WorldTrainerError.generation("The recording has no decodable video after its trim start.")
            }

            let eventURL = directory.appendingPathComponent(manifest.eventFile)
            var cursor = try EventFile.EventCursor(url: eventURL)
            var accumulator = ActionAccumulator(region: manifest.globalRect ?? .zero)
            let startNanos = try hostNanos(manifest, seconds: start)
            while let event = cursor.peek(), event.timestampNanos <= startNanos {
                accumulator.consume(event)
                _ = cursor.next()
            }
            accumulator.endTick()

            var decoded: [[UInt8]] = []
            var actions: [[Float]] = []
            try VideoFrames.streamRGB(
                info: info, start: start, end: end, fps: fps, width: width, height: height,
                onFrame: { index, data in
                    decoded.append([UInt8](data))
                    let frameEnd = min(end, start + Double(index + 1) / fps)
                    let endNanos = try hostNanos(manifest, seconds: frameEnd)
                    while let event = cursor.peek(), event.timestampNanos < endNanos {
                        accumulator.consume(event)
                        _ = cursor.next()
                    }
                    actions.append(accumulator.actionValues())
                    accumulator.endTick()
                })

            guard decoded.count >= 2 else {
                throw WorldTrainerError.generation(
                    "The recording is too short to seed a Run session. Choose a longer recording.")
            }
            return GenerationSeed(
                frames: decoded,
                actions: actions,
                recordingName: name,
                startSeconds: start,
                capabilities: capabilities)
        }.value
    }

    private static func hostNanos(_ manifest: RecordingManifest, seconds: Double) throws -> UInt64 {
        let nanos = seconds * 1_000_000_000
        guard seconds.isFinite, seconds >= 0, nanos < Double(UInt64.max - manifest.hostStartNanos) else {
            throw WorldTrainerError.generation("The recording contains a timestamp outside the supported range.")
        }
        return manifest.hostStartNanos + UInt64(nanos)
    }
}

/// One live generation session.
///
/// The engine owns the MLX model, its persistent recurrent memory and the
/// current frame. It is confined to the generation queue and never shared.
/// Nothing here bounds how many frames may be generated.
final class GenerationEngine {
    let configuration: WorldModelConfiguration
    let capabilities: ActionCapabilities
    let seedRecordingName: String
    let warmupFrameCount: Int

    private let model: WorldModel
    private var hidden: [MLXArray]
    private var current: MLXArray
    private(set) var generatedFrameCount = 0
    private(set) var lastStepSeconds: Double = 0
    private(set) var averageStepSeconds: Double = 0

    init(
        checkpoint: TrainingCheckpoint,
        checkpointDirectory: URL,
        seed: GenerationSeed,
        capabilities: ActionCapabilities
    ) throws {
        configuration = checkpoint.configuration
        self.capabilities = capabilities
        seedRecordingName = seed.recordingName

        let model = WorldModel(configuration: checkpoint.configuration)
        try CheckpointStore.restoreWeights(model, from: checkpointDirectory)
        model.train(false)
        self.model = model

        let count = min(seed.frames.count, seed.actions.count)
        guard count >= 2 else {
            throw WorldTrainerError.generation("The Run seed contained too few frames.")
        }
        let width = checkpoint.configuration.width
        let height = checkpoint.configuration.height

        var state = model.emptyHidden(batchSize: 1)
        // Teacher-forced warm-up: every pair (frame, action) advances the memory
        // exactly the way it advanced during training.
        for index in 0..<(count - 1) {
            let image = Self.tensor(fromRGB: seed.frames[index], width: width, height: height)
            let action = MLXArray(seed.actions[index], [1, ActionLayout.count])
            let (predicted, next) = model.step(image: image, action: action, hidden: state)
            state = next
            eval(predicted)
        }
        warmupFrameCount = count - 1
        hidden = state
        current = Self.tensor(fromRGB: seed.frames[count - 1], width: width, height: height)
    }

    /// Advances the world by one frame for the given control row.
    /// Returns the generated frame as packed RGB bytes.
    func step(action: [Float]) -> [UInt8] {
        let started = CFAbsoluteTimeGetCurrent()
        let actionArray = MLXArray(action, [1, ActionLayout.count])
        let (predicted, next) = model.step(image: current, action: actionArray, hidden: hidden)
        let rgb = round((predicted + 1) * 127.5).asType(.uint8)
        let bytes = rgb.asArray(UInt8.self)
        hidden = next
        current = predicted
        generatedFrameCount += 1
        let elapsed = CFAbsoluteTimeGetCurrent() - started
        lastStepSeconds = elapsed
        averageStepSeconds = averageStepSeconds == 0 ? elapsed : averageStepSeconds * 0.9 + elapsed * 0.1
        return bytes
    }

    static func tensor(fromRGB bytes: [UInt8], width: Int, height: Int) -> MLXArray {
        let raw = MLXArray(bytes, [1, height, width, 3])
        return raw.asType(.float32) / 127.5 - 1
    }
}
