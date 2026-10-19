import Foundation
import CoreGraphics

enum ModelResolution: String, Codable, CaseIterable, Hashable, Sendable, Identifiable {
    case compact = "128 × 80"
    case balanced = "160 × 104"
    case detailed = "192 × 120"

    var id: String { rawValue }

    var width: Int {
        switch self {
        case .compact: return 128
        case .balanced: return 160
        case .detailed: return 192
        }
    }

    var height: Int {
        switch self {
        case .compact: return 80
        case .balanced: return 104
        case .detailed: return 120
        }
    }

    var label: String { rawValue }

    var detail: String {
        switch self {
        case .compact: return "Fastest generation, coarsest detail"
        case .balanced: return "Recommended balance for 15–30 FPS"
        case .detailed: return "Sharpest output, slower training and generation"
        }
    }
}

/// Everything that changes the learned network or the training procedure.
/// Editing any of these fields invalidates exact checkpoint resume.
struct WorldModelConfiguration: Codable, Hashable, Sendable {
    var resolution: ModelResolution = .balanced
    var targetFPS: Double = 20
    var sequenceLength: Int = 12
    var baseChannels: Int = 32
    var recurrentChannels: Int = 96
    var actionEmbeddingChannels: Int = 24
    var learningRate: Double = 0.0004
    var weightDecay: Double = 0.0001
    var batchSize: Int = 2
    var epochs: Int = 10
    var seed: UInt64 = 1
    var gradientClip: Double = 1.0
    var validationFraction: Double = 0.1
    var trainingLossWeight: Double = 1.0
    var temporalConsistencyWeight: Double = 0.25
    var edgeLossWeight: Double = 0.1

    static let defaults = WorldModelConfiguration()

    var width: Int { resolution.width }
    var height: Int { resolution.height }

    var architectureFingerprint: String {
        [resolution.rawValue, "\(baseChannels)", "\(recurrentChannels)", "\(actionEmbeddingChannels)"].joined(separator: "|")
    }

    func validationIssues() -> [String] {
        var issues: [String] = []
        if targetFPS < 5 || targetFPS > 60 { issues.append("Target FPS must be between 5 and 60.") }
        if sequenceLength < 4 || sequenceLength > 32 { issues.append("Sequence length must be between 4 and 32 frames.") }
        if baseChannels < 16 || baseChannels > 64 { issues.append("Base channels must be between 16 and 64.") }
        if recurrentChannels < 32 || recurrentChannels > 192 { issues.append("Recurrent channels must be between 32 and 192.") }
        if actionEmbeddingChannels < 8 || actionEmbeddingChannels > 96 { issues.append("Action embedding must be between 8 and 96.") }
        if learningRate <= 0 || learningRate > 0.02 { issues.append("Learning rate must be greater than 0 and at most 0.02.") }
        if weightDecay < 0 || weightDecay > 0.1 { issues.append("Weight decay must be between 0 and 0.1.") }
        if batchSize < 1 || batchSize > 8 { issues.append("Batch size must be between 1 and 8.") }
        if epochs < 1 || epochs > 500 { issues.append("Epochs must be between 1 and 500.") }
        if gradientClip <= 0 || gradientClip > 100 { issues.append("Gradient clip must be between 0 and 100.") }
        if validationFraction < 0 || validationFraction > 0.5 { issues.append("Validation fraction must be between 0 and 0.5.") }
        if temporalConsistencyWeight < 0 || temporalConsistencyWeight > 4 { issues.append("Temporal loss weight must be between 0 and 4.") }
        if edgeLossWeight < 0 || edgeLossWeight > 4 { issues.append("Edge loss weight must be between 0 and 4.") }
        return issues
    }

    func estimatedParameterCount() -> Int {
        let c1 = baseChannels
        let c2 = baseChannels * 2
        let c3 = recurrentChannels
        let action = actionEmbeddingChannels
        let encoder = 3 * c1 * 9 + c1 * c2 * 9 + c2 * c3 * 9
        let cell = (c3 + action) * (2 * c3) * 9 + c3 * (2 * c3) * 9
            + (c3 + action) * c3 * 9 + c3 * c3 * 9
        let decoder = (c3 + c2) * c2 * 9 + (c2 + c1) * c1 * 9 + c1 * 24 * 9 + 24 * 3 * 9
        let actionMLP = ActionLayout.count * 96 + 96 * action
        return encoder + cell + decoder + actionMLP + 4_000
    }
}

enum ModelCheckpointKind: String, Codable, Hashable, Sendable {
    case periodic
    case epoch
    case paused
    case cancelled
    case best
}

struct TrainingCheckpoint: Codable, Hashable, Identifiable, Sendable {
    var id: UUID
    var modelID: UUID
    var createdAt: Date
    var step: Int
    var epoch: Int
    var sampleCursor: Int
    var epochSampleCount: Int
    var trainingLoss: Float?
    var validationLoss: Float?
    var bestValidationLoss: Float?
    var configurationFingerprint: String
    var datasetFingerprint: String
    var datasetRecordingIDs: [UUID]
    var trainingRecordingIDs: [UUID]
    var validationRecordingIDs: [UUID]
    var configuration: WorldModelConfiguration
    var seed: UInt64
    var kind: ModelCheckpointKind
    var parameterCount: Int

    var label: String {
        "Step \(step) · Epoch \(epoch)"
    }
}

struct WorldModelRecord: Codable, Hashable, Identifiable, Sendable {
    var id: UUID
    var name: String
    var createdAt: Date
    var updatedAt: Date
    var configuration: WorldModelConfiguration
    var recordingIDs: [UUID]
    var notes: String = ""
    var trainedCheckpointID: UUID?
    var trainedAt: Date?

    var checkpointSummary: String {
        guard let trainedCheckpointID else { return "Untrained" }
        return "Ready · \(trainedCheckpointID.uuidString.prefix(8))"
    }

    var isTrained: Bool { trainedCheckpointID != nil }
}

struct ModelsState: Codable, Sendable {
    var models: [WorldModelRecord] = []
}

struct DatasetCacheManifest: Codable, Sendable {
    struct RecordingEntry: Codable, Sendable {
        var recordingID: UUID
        var name: String
        var frameOffset: Int
        var frameCount: Int
        var actionOffset: Int
        var actionCount: Int
        var trainFrameStart: Int
        var trainFrameCount: Int
        var validationFrameStart: Int
        var validationFrameCount: Int
    }

    var fingerprint: String
    var createdAt: Date
    var configuration: WorldModelConfiguration
    var recordingIDs: [UUID]
    var recordings: [RecordingEntry]
    var totalFrames: Int
    var totalActions: Int
    var bytesPerFrame: Int
    var bytesPerAction: Int

    var trainingRecordingIDs: [UUID] {
        recordings.filter { $0.trainFrameCount > 0 }.map(\.recordingID)
    }
    var validationRecordingIDs: [UUID] {
        recordings.filter { $0.validationFrameCount > 0 }.map(\.recordingID)
    }
    var trainingFrameCount: Int { recordings.reduce(0) { $0 + $1.trainFrameCount } }
    var validationFrameCount: Int { recordings.reduce(0) { $0 + $1.validationFrameCount } }
}
