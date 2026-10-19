import Foundation
import SwiftUI

/// Drives `TrainingWorker` from the interface. The worker owns every MLX array;
/// this coordinator only publishes value snapshots on the main actor.
@MainActor
final class TrainingCoordinator: ObservableObject {
    @Published private(set) var progress = TrainingProgress()
    @Published private(set) var isRunning = false
    @Published private(set) var activeModelID: UUID?

    private let control = TrainingControl()
    private var checkpointHandler: ((TrainingCheckpoint) -> Void)?

    var canPause: Bool { isRunning && progress.phase.isBusy }

    var statusLine: String { progress.message }

    func start(
        model: WorldModelRecord,
        configuration: WorldModelConfiguration,
        sources: [DatasetRecordingSource],
        directories: AppDirectories,
        resume: Bool,
        memoryLimitGB: Double,
        cacheLimitGB: Double,
        onCheckpoint: @escaping (TrainingCheckpoint) -> Void
    ) {
        guard !isRunning else { return }
        checkpointHandler = onCheckpoint
        isRunning = true
        activeModelID = model.id
        control.set(.run)

        let request = TrainingRequest(
            model: model,
            configuration: configuration,
            sources: sources,
            directories: directories,
            resume: resume,
            memoryLimitGB: memoryLimitGB,
            cacheLimitGB: cacheLimitGB)

        let publish: @Sendable (TrainingProgress) -> Void = { [weak self] snapshot in
            Task { @MainActor in self?.handle(snapshot) }
        }
        let saved: @Sendable (TrainingCheckpoint) -> Void = { [weak self] checkpoint in
            Task { @MainActor in self?.checkpointHandler?(checkpoint) }
        }

        TrainingWorker.queue.async {
            TrainingWorker.run(request, control: self.control, publish: publish, checkpointSaved: saved)
        }
    }

    func pause() {
        guard isRunning else { return }
        control.set(.pause)
        progress.message = "Pausing after the current step…"
    }

    func cancel() {
        guard isRunning else { return }
        control.set(.cancel)
        progress.message = "Stopping after the current step…"
    }

    func clear() {
        guard !isRunning else { return }
        progress = TrainingProgress()
        activeModelID = nil
    }

    private func handle(_ snapshot: TrainingProgress) {
        progress = snapshot
        switch snapshot.phase {
        case .paused, .cancelled, .complete, .failed, .idle:
            isRunning = false
        default:
            isRunning = true
        }
    }
}
