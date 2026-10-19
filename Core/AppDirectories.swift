import Foundation

enum WorldTrainerError: LocalizedError, Sendable {
    case storage(String)
    case invalidData(String)
    case training(String)
    case generation(String)
    case importFailure(String)

    var errorDescription: String? {
        switch self {
        case .storage(let message), .invalidData(let message), .training(let message),
             .generation(let message), .importFailure(let message):
            return message
        }
    }
}

/// Filesystem layout for one WorldTrainer data root.
///
/// ```
/// <root>/
///   library.json
///   Recordings/<catalog id>/
///   models.json
///   Models/<model id>/model.json
///   Models/<model id>/checkpoints/<checkpoint id>/
///   TrainingCache/<model id>/<fingerprint>/
/// ```
struct AppDirectories: Hashable, Sendable {
    var root: URL

    init(root: URL) {
        self.root = root.standardizedFileURL
    }

    static var defaultRoot: URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support")
        return support.appendingPathComponent("WorldTrainer", isDirectory: true)
    }

    var libraryFile: URL { root.appendingPathComponent("library.json") }
    var modelsFile: URL { root.appendingPathComponent("models.json") }
    var recordingsDirectory: URL { root.appendingPathComponent("Recordings", isDirectory: true) }
    var modelsDirectory: URL { root.appendingPathComponent("Models", isDirectory: true) }
    var trainingCacheDirectory: URL { root.appendingPathComponent("TrainingCache", isDirectory: true) }

    func recordingDirectory(id: UUID) -> URL {
        recordingsDirectory.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    func modelDirectory(id: UUID) -> URL {
        modelsDirectory.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    func modelFile(id: UUID) -> URL {
        modelDirectory(id: id).appendingPathComponent("model.json")
    }

    func checkpointDirectory(modelID: UUID, checkpointID: UUID) -> URL {
        modelDirectory(id: modelID)
            .appendingPathComponent("checkpoints", isDirectory: true)
            .appendingPathComponent(checkpointID.uuidString, isDirectory: true)
    }

    func trainingCache(modelID: UUID, fingerprint: String) -> URL {
        trainingCacheDirectory
            .appendingPathComponent(modelID.uuidString, isDirectory: true)
            .appendingPathComponent(fingerprint, isDirectory: true)
    }

    func prepare() throws {
        let manager = FileManager.default
        for directory in [root, recordingsDirectory, modelsDirectory, trainingCacheDirectory] {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }

    func storageBytes() -> Int64 {
        Self.directoryBytes(root)
    }

    static func directoryBytes(_ url: URL) -> Int64 {
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileAllocatedSizeKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles], errorHandler: nil
        ) else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: Set(keys)),
                  values.isRegularFile == true else { continue }
            total += Int64(values.fileAllocatedSize ?? values.fileSize ?? 0)
        }
        return total
    }

    static func fileBytes(_ url: URL) -> Int64 {
        (try? url.resourceValues(forKeys: [.fileAllocatedSizeKey, .fileSizeKey]).fileAllocatedSize)
            .flatMap { $0.map(Int64.init) }
            ?? (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).flatMap { $0.map(Int64.init) }
            ?? 0
    }
}
