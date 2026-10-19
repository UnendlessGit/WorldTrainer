import Foundation
import MLX
import SwiftUI

enum AppTab: String, CaseIterable, Identifiable, Hashable {
    case library
    case models
    case training
    case run
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .library: return "Library"
        case .models: return "World Models"
        case .training: return "Training"
        case .run: return "Run"
        case .settings: return "Settings"
        }
    }

    var systemImage: String {
        switch self {
        case .library: return "square.stack.3d.up"
        case .models: return "brain"
        case .training: return "chart.line.uptrend.xyaxis"
        case .run: return "play.rectangle.on.rectangle"
        case .settings: return "gearshape"
        }
    }
}

/// One place that owns the stores so every tab sees the same state.
@MainActor
final class AppStore: ObservableObject {
    let settings: SettingsStore
    let library: LibraryStore
    let models: ModelsStore
    let training: TrainingCoordinator
    let run: GenerationCoordinator

    @Published var selectedTab: AppTab = .library
    @Published var libraryFolderSelection: UUID?
    @Published var libraryRecordingSelection: UUID?
    @Published var modelsSelection: UUID?
    @Published var trainingModelID: UUID?
    @Published var runModelID: UUID?
    @Published var runError: String?

    init() {
        let settings = SettingsStore()
        self.settings = settings
        let directories = settings.directories
        library = LibraryStore(directories: directories)
        models = ModelsStore(directories: directories)
        training = TrainingCoordinator()
        run = GenerationCoordinator()
        run.targetFPS = settings.preferences.defaultRunFPS
    }

    var directories: AppDirectories { settings.directories }

    /// Re-points every store when the storage location changes.
    func applyStorageLocationChange() {
        library.updateDirectories(directories)
        models.updateDirectories(directories)
    }

    func selectTab(_ tab: AppTab) {
        if selectedTab == .run, tab != .run {
            // Leaving Run releases every held control and stops generating, so
            // nothing stays pressed and no GPU time is spent off screen.
            run.stop()
        }
        selectedTab = tab
    }

    // MARK: - Training

    /// Validates the selection and hands it to the training worker.
    /// Returns a message when the run could not start.
    @discardableResult
    func startTraining(modelID: UUID, resume: Bool) -> String? {
        guard let model = models.model(id: modelID) else { return "Select a model first." }
        guard !training.isRunning else { return "A training run is already active." }
        let configuration = model.configuration
        let issues = configuration.validationIssues()
        guard issues.isEmpty else { return issues.joined(separator: " ") }
        let sources = models.sources(for: model, library: library).filter { $0.manifest.isStructurallyValid }
        guard !sources.isEmpty else {
            return "This model has no usable recordings selected. Choose recordings in the World Models tab."
        }
        if resume,
           CheckpointStore.latest(
               modelID: model.id,
               configurationFingerprint: configuration.architectureFingerprint,
               datasetFingerprint: DatasetBuilder.fingerprint(configuration: configuration, sources: sources),
               directories: directories) == nil {
            return "No resumable checkpoint matches this configuration and dataset yet. Start training instead."
        }
        training.start(
            model: model,
            configuration: configuration,
            sources: sources,
            directories: directories,
            resume: resume,
            memoryLimitGB: settings.preferences.memoryLimitGB,
            cacheLimitGB: settings.preferences.cacheLimitGB,
            onCheckpoint: { [weak self] checkpoint in
                guard let self else { return }
                models.refreshCheckpoints(modelID: checkpoint.modelID)
                // Any checkpoint can drive a Run session, so the newest one
                // becomes the model's active checkpoint immediately.
                models.markTrained(modelID: checkpoint.modelID, checkpoint: checkpoint)
            })
        return nil
    }

    // MARK: - Run

    /// Prepares a Run session: loads the trained checkpoint, decodes a short
    /// warm-up window from a recording, then starts generating.
    func startRun(modelID: UUID) async {
        runError = nil
        guard let model = models.model(id: modelID) else {
            runError = "Select a trained model."
            return
        }
        let checkpointID: UUID
        if let id = model.trainedCheckpointID, models.checkpoint(modelID: model.id, checkpointID: id) != nil {
            checkpointID = id
        } else if let newest = models.checkpoints(for: model.id).first {
            checkpointID = newest.id
            models.markTrained(modelID: model.id, checkpoint: newest)
        } else {
            runError = "\(model.name) has no saved checkpoint yet. Train it first."
            return
        }
        guard let loaded = models.checkpoint(modelID: model.id, checkpointID: checkpointID) else {
            runError = "The checkpoint for \(model.name) is missing on disk."
            return
        }
        let sources = models.sources(for: model, library: library)
        guard let seedSource = sources.first(where: { $0.manifest.isStructurallyValid }) else {
            runError = "This model has no usable recording to seed generation from."
            return
        }
        let capabilities = models.capabilities(for: model, library: library)
        guard !capabilities.isEmpty else {
            runError = "The recordings behind this model contain no input events, so there is nothing to control."
            return
        }
        run.prepare()
        do {
            let seed = try await GenerationSeedBuilder.build(
                directory: seedSource.directory,
                manifest: seedSource.manifest,
                configuration: loaded.checkpoint.configuration,
                warmupFrames: settings.preferences.runWarmupFrames,
                capabilities: capabilities)
            run.start(
                model: model,
                checkpoint: loaded.checkpoint,
                checkpointDirectory: loaded.directory,
                seed: seed,
                capabilities: capabilities,
                memoryLimitGB: settings.preferences.memoryLimitGB)
        } catch {
            run.fail(error.localizedDescription)
            runError = error.localizedDescription
        }
    }

    // MARK: - Diagnostics

    func diagnostics() -> [(String, String)] {
        let processInfo = ProcessInfo.processInfo
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return [
            ("App", "WorldTrainer \(version) (\(build))"),
            ("macOS", processInfo.operatingSystemVersionString),
            ("Chip", Self.sysctlString("machdep.cpu.brand_string") ?? "Apple Silicon"),
            ("Memory", ByteCountFormatter.string(fromByteCount: Int64(processInfo.physicalMemory), countStyle: .memory)),
            ("Compute", "Metal GPU · \(MLX.GPU.deviceInfo().architecture)"),
            ("MLX memory limit", ByteCountFormatter.string(fromByteCount: Int64(Memory.memoryLimit), countStyle: .memory)),
            ("MLX cache limit", ByteCountFormatter.string(fromByteCount: Int64(Memory.cacheLimit), countStyle: .memory)),
            ("Active memory", ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .memory)),
            ("Peak memory", ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .memory)),
            ("Data root", directories.root.path),
        ]
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer)
    }
}
