import CoreGraphics
import Foundation
import SwiftUI

enum RunPhase: Equatable {
    case idle
    case preparing
    case running
    case stopped
    case failed

    var label: String {
        switch self {
        case .idle: return "Idle"
        case .preparing: return "Preparing"
        case .running: return "Generating"
        case .stopped: return "Stopped"
        case .failed: return "Failed"
        }
    }
}

/// What a Run session was started from, shown in the interface so the source of
/// every generated frame is explicit.
struct RunDetail: Equatable, Sendable {
    var modelName = ""
    var checkpointLabel = ""
    var seedRecording = ""
    var seedStartSeconds: Double = 0
    var warmupFrames = 0
    var parameterCount = 0
    var recordingCount = 0
    var datasetFrames = 0
    var resolution = ""
    var startedAt: Date?
    var capabilityLines: [String] = []
}

@MainActor
final class GenerationCoordinator: ObservableObject {
    @Published private(set) var phase: RunPhase = .idle
    @Published private(set) var frame: CGImage?
    @Published private(set) var frameIndex = 0
    @Published private(set) var measuredFPS: Double = 0
    @Published private(set) var headroomFPS: Double = 0
    @Published private(set) var stepMilliseconds: Double = 0
    @Published private(set) var elapsedSeconds: Double = 0
    @Published private(set) var error: String?
    @Published private(set) var detail = RunDetail()

    @Published var targetFPS: Double = 20 {
        didSet { session?.setTargetFPS(targetFPS) }
    }

    let capture = RunInputCapture()
    private var session: GenerationSession?

    var isActive: Bool { phase == .running || phase == .preparing }

    func start(
        model: WorldModelRecord,
        checkpoint: TrainingCheckpoint,
        checkpointDirectory: URL,
        seed: GenerationSeed,
        capabilities: ActionCapabilities,
        memoryLimitGB: Double
    ) {
        stop(keepFrame: true)
        error = nil
        frameIndex = 0
        measuredFPS = 0
        headroomFPS = 0
        stepMilliseconds = 0
        elapsedSeconds = 0
        phase = .preparing
        detail = RunDetail(
            modelName: model.name,
            checkpointLabel: "\(checkpoint.label) · \(checkpoint.kind.rawValue)",
            seedRecording: seed.recordingName,
            seedStartSeconds: seed.startSeconds,
            warmupFrames: seed.warmupFrameCount,
            parameterCount: checkpoint.parameterCount,
            recordingCount: checkpoint.datasetRecordingIDs.count,
            datasetFrames: checkpoint.epochSampleCount,
            resolution: "\(checkpoint.configuration.width) × \(checkpoint.configuration.height)",
            startedAt: nil,
            capabilityLines: capabilities.summaryLines)

        let session = GenerationSession(
            inputBox: capture.box,
            capabilities: capabilities,
            region: CodableRect(x: 0, y: 0, width: 1, height: 1),
            publish: { [weak self] snapshot in
                Task { @MainActor in self?.handle(snapshot) }
            })
        session.setTargetFPS(targetFPS)
        self.session = session
        capture.releaseAll()
        capture.beginReporting()
        session.prepare(
            checkpoint: checkpoint,
            checkpointDirectory: checkpointDirectory,
            seed: seed,
            memoryLimitGB: memoryLimitGB,
            targetFPS: targetFPS)
    }

    func stop(keepFrame: Bool = true) {
        capture.endReporting()
        capture.releaseAll()
        session?.stop()
        session = nil
        if phase == .running || phase == .preparing {
            phase = .stopped
        }
        if !keepFrame { frame = nil }
    }

    func clearError() {
        error = nil
        if phase == .failed { phase = .idle }
    }

    /// Shows that a session is being assembled (checkpoint load and warm-up).
    func prepare() {
        phase = .preparing
        error = nil
    }

    /// Reports a failure that happened before a session could be created.
    func fail(_ message: String) {
        phase = .failed
        error = message
    }

    private func handle(_ snapshot: GenerationSnapshot) {
        switch snapshot.kind {
        case .ready:
            phase = .running
            detail.startedAt = Date()
        case .frame:
            if let image = FrameCodec.image(fromRGB: snapshot.bytes, width: snapshot.width, height: snapshot.height) {
                frame = image
            }
            frameIndex = snapshot.frameIndex
            elapsedSeconds = snapshot.elapsedSeconds
            if snapshot.measuredFPS > 0 { measuredFPS = snapshot.measuredFPS }
            if snapshot.headroomFPS > 0 { headroomFPS = snapshot.headroomFPS }
            if snapshot.stepMilliseconds > 0 { stepMilliseconds = snapshot.stepMilliseconds }
        case .finished:
            if phase == .running { phase = .stopped }
        case .failed(let message):
            phase = .failed
            error = message
            capture.endReporting()
        }
    }
}
