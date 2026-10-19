import Foundation
import MLX

/// Deterministic PRNG so an epoch's sample order can be reconstructed exactly
/// after a pause, restart, or crash.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed &+ 0x9E37_79B9_7F4A_7C15
    }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

struct TrainingSample: Hashable, Sendable {
    var recordingID: UUID
    var startFrame: Int
}

struct TrainingSchedule: Sendable {
    let samples: [TrainingSample]
    let batchSize: Int

    init(manifest: DatasetCacheManifest, sequenceLength: Int, batchSize: Int, epoch: Int, seed: UInt64) {
        var result: [TrainingSample] = []
        for entry in manifest.recordings where entry.trainFrameCount > sequenceLength {
            let first = entry.trainFrameStart
            let last = entry.trainFrameStart + entry.trainFrameCount - sequenceLength - 1
            if last >= first {
                for start in first...last {
                    result.append(TrainingSample(recordingID: entry.recordingID, startFrame: start))
                }
            }
        }
        var generator = SplitMix64(seed: seed &+ UInt64(epoch) &* 0x9E37_79B9)
        result.shuffle(using: &generator)
        samples = result
        self.batchSize = batchSize
    }

    var stepCount: Int {
        max(1, Int(ceil(Double(samples.count) / Double(max(1, batchSize)))))
    }

    func batch(atStep step: Int) -> ArraySlice<TrainingSample> {
        let lower = min(samples.count, step * batchSize)
        let upper = min(samples.count, lower + batchSize)
        return samples[lower..<upper]
    }

    func validationSamples(manifest: DatasetCacheManifest, sequenceLength: Int, limit: Int = 64) -> [TrainingSample] {
        var candidates: [TrainingSample] = []
        for entry in manifest.recordings where entry.validationFrameCount > sequenceLength {
            let first = entry.validationFrameStart
            let last = entry.validationFrameStart + entry.validationFrameCount - sequenceLength - 1
            if last >= first {
                for start in first...last {
                    candidates.append(TrainingSample(recordingID: entry.recordingID, startFrame: start))
                }
            }
        }
        guard candidates.count > limit else { return candidates }
        let stride = Double(candidates.count) / Double(limit)
        return (0..<limit).map { candidates[min(candidates.count - 1, Int(Double($0) * stride))] }
    }
}

/// Memory-mapped dataset reader. Sequences never cross a recording boundary.
final class DatasetReader {
    private let framesData: Data
    private let actionsData: Data
    private let manifest: DatasetCacheManifest

    init(directory: URL, manifest: DatasetCacheManifest) throws {
        self.manifest = manifest
        framesData = try Data(contentsOf: directory.appendingPathComponent("frames.bin"), options: .mappedIfSafe)
        actionsData = try Data(contentsOf: directory.appendingPathComponent("actions.bin"), options: .mappedIfSafe)
        let expectedFrames = manifest.totalFrames * manifest.bytesPerFrame
        let expectedActions = manifest.totalActions * manifest.bytesPerAction
        guard framesData.count >= expectedFrames, actionsData.count >= expectedActions else {
            throw WorldTrainerError.invalidData("The dataset cache is truncated. Remove it and train again.")
        }
    }

    /// Load a batch of `[B,T+1,H,W,3]` frames and `[B,T,146]` actions.
    func batch(samples: [TrainingSample], sequenceLength: Int) -> (frames: MLXArray, actions: MLXArray) {
        let batch = samples.count
        let height = manifest.configuration.height
        let width = manifest.configuration.width
        let frameValues = (sequenceLength + 1) * height * width * 3
        let actionValues = ActionLayout.count
        var frames = [Float](repeating: 0, count: batch * frameValues)
        var actions = [Float](repeating: 0, count: batch * sequenceLength * actionValues)

        framesData.withUnsafeBytes { rawFrames in
            guard let frameBase = rawFrames.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            actionsData.withUnsafeBytes { rawActions in
                guard let actionBase = rawActions.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
                for (batchIndex, sample) in samples.enumerated() {
                    let frameStart = sample.startFrame * manifest.bytesPerFrame
                    let frameTarget = batchIndex * frameValues
                    for valueIndex in 0..<frameValues {
                        let byte = frameBase[frameStart + valueIndex]
                        frames[frameTarget + valueIndex] = Float(byte) / 127.5 - 1
                    }
                    let actionStart = sample.startFrame * ActionLayout.count
                    let actionTarget = batchIndex * sequenceLength * actionValues
                    for valueIndex in 0..<(sequenceLength * actionValues) {
                        actions[actionTarget + valueIndex] = actionBase[actionStart + valueIndex]
                    }
                }
            }
        }
        return (
            MLXArray(frames, [batch, sequenceLength + 1, height, width, 3]),
            MLXArray(actions, [batch, sequenceLength, ActionLayout.count]))
    }
}
