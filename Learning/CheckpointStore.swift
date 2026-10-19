import Foundation
import MLX
import MLXNN

enum CheckpointStore {
    static func save(
        checkpoint: TrainingCheckpoint,
        model: WorldModel,
        optimizer: ResumableAdamW,
        directories: AppDirectories
    ) throws -> TrainingCheckpoint {
        let directory = directories.checkpointDirectory(modelID: checkpoint.modelID, checkpointID: checkpoint.id)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try MLX.save(arrays: weights, url: directory.appendingPathComponent("weights.safetensors"))
        try MLX.save(arrays: optimizer.arrays(), url: directory.appendingPathComponent("optimizer.safetensors"))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(checkpoint).write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
        try prune(modelID: checkpoint.modelID, directories: directories)
        return checkpoint
    }

    static func list(modelID: UUID, directories: AppDirectories) -> [TrainingCheckpoint] {
        let root = directories.modelDirectory(id: modelID).appendingPathComponent("checkpoints", isDirectory: true)
        guard let urls = try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return urls.compactMap { url -> TrainingCheckpoint? in
            guard let data = try? Data(contentsOf: url.appendingPathComponent("manifest.json")) else { return nil }
            return try? decoder.decode(TrainingCheckpoint.self, from: data)
        }.sorted { $0.step > $1.step }
    }

    static func latest(
        modelID: UUID,
        configurationFingerprint: String,
        datasetFingerprint: String,
        directories: AppDirectories
    ) -> (checkpoint: TrainingCheckpoint, directory: URL)? {
        guard let checkpoint = list(modelID: modelID, directories: directories).first(where: {
            $0.configurationFingerprint == configurationFingerprint && $0.datasetFingerprint == datasetFingerprint
        }) else { return nil }
        return (checkpoint, directories.checkpointDirectory(modelID: modelID, checkpointID: checkpoint.id))
    }

    static func load(
        modelID: UUID,
        checkpointID: UUID,
        directories: AppDirectories
    ) -> (checkpoint: TrainingCheckpoint, directory: URL)? {
        let directory = directories.checkpointDirectory(modelID: modelID, checkpointID: checkpointID)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("manifest.json")),
              let checkpoint = try? decoder.decode(TrainingCheckpoint.self, from: data) else { return nil }
        return (checkpoint, directory)
    }

    static func delete(modelID: UUID, checkpointID: UUID, directories: AppDirectories) throws {
        let directory = directories.checkpointDirectory(modelID: modelID, checkpointID: checkpointID)
        try FileManager.default.removeItem(at: directory)
    }

    static func restoreWeights(_ model: WorldModel, from directory: URL) throws {
        let weights = try MLX.loadArrays(url: directory.appendingPathComponent("weights.safetensors"))
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        guard Set(expected.keys) == Set(weights.keys),
              expected.allSatisfy({ weights[$0.key]?.shape == $0.value.shape }) else {
            throw WorldTrainerError.invalidData("Checkpoint tensor names or shapes do not match the model configuration.")
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
    }

    static func restoreOptimizer(_ optimizer: ResumableAdamW, model: WorldModel, from directory: URL) throws {
        let state = try MLX.loadArrays(url: directory.appendingPathComponent("optimizer.safetensors"))
        try optimizer.restore(state, model: model)
    }

    private static func prune(modelID: UUID, directories: AppDirectories) throws {
        let all = list(modelID: modelID, directories: directories)
        guard all.count > 4 else { return }
        var keep = Set(all.prefix(3).map(\.id))
        if let best = all.filter({ $0.kind == .best || $0.validationLoss != nil }).min(by: {
            ($0.bestValidationLoss ?? $0.validationLoss ?? .infinity) < ($1.bestValidationLoss ?? $1.validationLoss ?? .infinity)
        }) {
            keep.insert(best.id)
        }
        for checkpoint in all where !keep.contains(checkpoint.id) {
            let directory = directories.checkpointDirectory(modelID: modelID, checkpointID: checkpoint.id)
            try? FileManager.default.removeItem(at: directory)
        }
    }
}
