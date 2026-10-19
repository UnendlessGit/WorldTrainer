import AppKit
import Foundation

/// Model definitions, their dataset selections and their checkpoints.
@MainActor
final class ModelsStore: ObservableObject {
    @Published private(set) var state = ModelsState()
    @Published private(set) var checkpoints: [UUID: [TrainingCheckpoint]] = [:]
    @Published var lastError: String?
    @Published var lastReport: String?

    private var directories: AppDirectories

    init(directories: AppDirectories) {
        self.directories = directories
        load()
        refreshAllCheckpoints()
    }

    func updateDirectories(_ directories: AppDirectories) {
        self.directories = directories
        load()
        refreshAllCheckpoints()
    }

    var models: [WorldModelRecord] {
        state.models.sorted { $0.updatedAt > $1.updatedAt }
    }

    var trainedModels: [WorldModelRecord] {
        models.filter { $0.isTrained }
    }

    func model(id: UUID) -> WorldModelRecord? {
        state.models.first { $0.id == id }
    }

    func checkpoints(for modelID: UUID) -> [TrainingCheckpoint] {
        checkpoints[modelID] ?? []
    }

    func checkpoint(modelID: UUID, checkpointID: UUID) -> (checkpoint: TrainingCheckpoint, directory: URL)? {
        CheckpointStore.load(modelID: modelID, checkpointID: checkpointID, directories: directories)
    }

    func checkpointCount(for modelID: UUID) -> Int {
        checkpoints[modelID]?.count ?? 0
    }

    func refreshAllCheckpoints() {
        var result: [UUID: [TrainingCheckpoint]] = [:]
        for model in state.models {
            result[model.id] = CheckpointStore.list(modelID: model.id, directories: directories)
        }
        checkpoints = result
    }

    func refreshCheckpoints(modelID: UUID) {
        checkpoints[modelID] = CheckpointStore.list(modelID: modelID, directories: directories)
    }

    // MARK: - Persistence

    private func load() {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: directories.modelsFile),
              let decoded = try? decoder.decode(ModelsState.self, from: data) else {
            state = ModelsState()
            return
        }
        state = decoded
    }

    private func save() {
        do {
            try directories.prepare()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(state).write(to: directories.modelsFile, options: .atomic)
        } catch {
            lastError = "The model index could not be saved: \(error.localizedDescription)"
        }
    }

    /// Writes one model's own descriptor. Kept separate from the index so
    /// editing a model does not rewrite every other model's file.
    private func writeModelFile(_ model: WorldModelRecord) {
        do {
            let directory = directories.modelDirectory(id: model.id)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(model).write(to: directories.modelFile(id: model.id), options: .atomic)
        } catch {
            lastError = "The model descriptor could not be saved: \(error.localizedDescription)"
        }
    }

    // MARK: - Editing

    @discardableResult
    func create(name: String, configuration: WorldModelConfiguration, recordingIDs: [UUID]) -> WorldModelRecord {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let model = WorldModelRecord(
            id: UUID(),
            name: trimmed.isEmpty ? "Model \(state.models.count + 1)" : trimmed,
            createdAt: Date(),
            updatedAt: Date(),
            configuration: configuration,
            recordingIDs: recordingIDs)
        state.models.append(model)
        save()
        writeModelFile(model)
        checkpoints[model.id] = []
        lastReport = "Created \(model.name)."
        return model
    }

    func update(_ model: WorldModelRecord, touch: Bool = true) {
        guard let index = state.models.firstIndex(where: { $0.id == model.id }) else { return }
        var updated = model
        if touch { updated.updatedAt = Date() }
        state.models[index] = updated
        save()
        writeModelFile(updated)
    }

    func rename(_ model: WorldModelRecord, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var updated = model
        updated.name = trimmed
        update(updated)
    }

    func setDatasets(_ recordingIDs: [UUID], for model: WorldModelRecord) {
        var updated = model
        updated.recordingIDs = recordingIDs
        update(updated)
    }

    func markTrained(modelID: UUID, checkpoint: TrainingCheckpoint) {
        guard var model = model(id: modelID) else { return }
        model.trainedCheckpointID = checkpoint.id
        model.trainedAt = Date()
        update(model)
        refreshCheckpoints(modelID: modelID)
    }

    func deleteCheckpoint(modelID: UUID, checkpointID: UUID) {
        do {
            try CheckpointStore.delete(modelID: modelID, checkpointID: checkpointID, directories: directories)
        } catch {
            lastError = "The checkpoint could not be removed: \(error.localizedDescription)"
        }
        if var model = model(id: modelID), model.trainedCheckpointID == checkpointID {
            model.trainedCheckpointID = nil
            model.trainedAt = nil
            update(model, touch: false)
        }
        refreshCheckpoints(modelID: modelID)
    }

    func delete(_ model: WorldModelRecord, removeFiles: Bool) {
        state.models.removeAll { $0.id == model.id }
        checkpoints[model.id] = nil
        save()
        guard removeFiles else { return }
        let modelDirectory = directories.modelDirectory(id: model.id)
        try? FileManager.default.removeItem(at: modelDirectory)
        let cacheDirectory = directories.trainingCacheDirectory.appendingPathComponent(model.id.uuidString)
        try? FileManager.default.removeItem(at: cacheDirectory)
    }

    func reveal(_ model: WorldModelRecord) {
        let directory = directories.modelDirectory(id: model.id)
        if FileManager.default.fileExists(atPath: directory.path) {
            NSWorkspace.shared.activateFileViewerSelecting([directory])
        } else {
            NSWorkspace.shared.activateFileViewerSelecting([directories.modelsDirectory])
        }
    }

    func revealCheckpoint(modelID: UUID, checkpointID: UUID) {
        NSWorkspace.shared.activateFileViewerSelecting([
            directories.checkpointDirectory(modelID: modelID, checkpointID: checkpointID)
        ])
    }

    // MARK: - Datasets

    /// Resolves a model's recording selection into dataset sources.
    func sources(for model: WorldModelRecord, library: LibraryStore) -> [DatasetRecordingSource] {
        model.recordingIDs.compactMap { id in
            guard let recording = library.recording(id: id), let manifest = recording.manifest else { return nil }
            return DatasetRecordingSource(
                recordingID: recording.id,
                name: recording.displayName,
                directory: library.directory(for: recording),
                manifest: manifest,
                videoDuration: recording.videoDuration)
        }
    }

    func capabilities(for model: WorldModelRecord, library: LibraryStore) -> ActionCapabilities {
        let recordings = model.recordingIDs.compactMap { library.recording(id: $0) }
        return ActionCapabilities.union(recordings.map(\.capabilities))
    }

    func cacheBytes(for modelID: UUID) -> Int64 {
        AppDirectories.directoryBytes(directories.trainingCacheDirectory.appendingPathComponent(modelID.uuidString))
    }
}
