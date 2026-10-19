import AppKit
import Foundation

struct ImportProgress: Sendable, Equatable {
    var message: String
    var fraction: Double
    var imported: Int
    var total: Int
}

private struct ImportResult: Sendable {
    var recordings: [LibraryRecording]
    var failures: [String]
    var skipped: Int
}

@MainActor
final class LibraryStore: ObservableObject {
    @Published private(set) var state = LibraryState()
    @Published var importProgress: ImportProgress?
    @Published var lastError: String?
    @Published var lastReport: String?
    @Published private(set) var isRefreshing = false

    private var directories: AppDirectories

    init(directories: AppDirectories) {
        self.directories = directories
        load()
    }

    func updateDirectories(_ directories: AppDirectories) {
        self.directories = directories
        load()
    }

    var folders: [LibraryFolder] { state.folders }
    var recordings: [LibraryRecording] { state.recordings }

    var usableRecordings: [LibraryRecording] {
        state.recordings.filter { $0.usability == .usable }
    }

    func recordings(in folderID: UUID?) -> [LibraryRecording] {
        state.recordings
            .filter { folderID == nil || $0.folderID == folderID }
            .sorted { $0.importedAt > $1.importedAt }
    }

    func recording(id: UUID) -> LibraryRecording? {
        state.recordings.first { $0.id == id }
    }

    func directory(for recording: LibraryRecording) -> URL {
        directories.recordingsDirectory.appendingPathComponent(recording.directoryName, isDirectory: true)
    }

    func thumbnailURL(for recording: LibraryRecording) -> URL? {
        guard let name = recording.manifest?.thumbnailFile else { return nil }
        return directory(for: recording).appendingPathComponent(name)
    }

    // MARK: - Persistence

    private func load() {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: directories.libraryFile),
              let decoded = try? decoder.decode(LibraryState.self, from: data) else {
            state = LibraryState()
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
            try encoder.encode(state).write(to: directories.libraryFile, options: .atomic)
        } catch {
            lastError = "The library index could not be saved: \(error.localizedDescription)"
        }
    }

    // MARK: - Folders

    func addFolder(named name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        state.folders.append(LibraryFolder(id: UUID(), name: trimmed, createdAt: Date()))
        save()
    }

    func renameFolder(_ folder: LibraryFolder, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let index = state.folders.firstIndex(where: { $0.id == folder.id }) else { return }
        state.folders[index].name = trimmed
        save()
    }

    /// Deleting a folder never deletes recordings. They move back to Unfiled.
    func deleteFolder(_ folder: LibraryFolder) {
        state.folders.removeAll { $0.id == folder.id }
        for index in state.recordings.indices where state.recordings[index].folderID == folder.id {
            state.recordings[index].folderID = nil
        }
        save()
    }

    func move(_ recording: LibraryRecording, to folderID: UUID?) {
        guard let index = state.recordings.firstIndex(where: { $0.id == recording.id }) else { return }
        state.recordings[index].folderID = folderID
        save()
    }

    // MARK: - Recording edits

    func rename(_ recording: LibraryRecording, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let index = state.recordings.firstIndex(where: { $0.id == recording.id }) else { return }
        state.recordings[index].displayName = trimmed
        save()
    }

    func delete(_ recording: LibraryRecording, removeFiles: Bool) {
        let directory = directory(for: recording)
        state.recordings.removeAll { $0.id == recording.id }
        save()
        if removeFiles {
            try? FileManager.default.removeItem(at: directory)
        }
    }

    func reveal(_ recording: LibraryRecording) {
        NSWorkspace.shared.activateFileViewerSelecting([directory(for: recording)])
    }

    /// Re-reads every package on disk so external edits and missing files are
    /// reported instead of silently drifting from the catalog.
    func refresh() {
        guard !isRefreshing else { return }
        isRefreshing = true
        lastError = nil
        let directories = self.directories
        let snapshot = state.recordings
        Task {
            var updated = snapshot
            var missing = 0
            for index in updated.indices {
                let packageDirectory = directories.recordingsDirectory
                    .appendingPathComponent(updated[index].directoryName, isDirectory: true)
                guard FileManager.default.fileExists(atPath: packageDirectory.path) else {
                    missing += 1
                    updated[index].issues = [.error("missing-package",
                        "The package folder is missing from the library directory. Remove this entry or restore the folder.")]
                    continue
                }
                let inspection = await RecordingInspector.inspect(packageDirectory: packageDirectory)
                updated[index].manifest = inspection.manifest
                updated[index].issues = inspection.issues
                updated[index].eventSummary = inspection.eventSummary
                updated[index].packageBytes = inspection.packageBytes
                updated[index].videoWidth = inspection.videoWidth
                updated[index].videoHeight = inspection.videoHeight
                updated[index].videoDuration = inspection.videoDuration
            }
            state.recordings = updated
            save()
            isRefreshing = false
            lastReport = missing == 0
                ? "Rechecked \(updated.count) recording\(updated.count == 1 ? "" : "s")."
                : "Rechecked \(updated.count) recordings. \(missing) package folder\(missing == 1 ? " is" : "s are") missing."
        }
    }

    // MARK: - Import

    func importPackages(from urls: [URL], into folderID: UUID?) async {
        guard importProgress == nil else { return }
        importProgress = ImportProgress(message: "Looking for recording packages…", fraction: 0, imported: 0, total: 0)
        lastError = nil
        lastReport = nil

        let destinations = directories
        let existingIDs = Set(state.recordings.map(\.id))
        let existingNames = Set(state.recordings.map(\.directoryName))
        let report: @Sendable (String, Double, Int, Int) -> Void = { message, fraction, imported, total in
            Task { @MainActor [weak self] in
                self?.importProgress = ImportProgress(message: message, fraction: fraction, imported: imported, total: total)
            }
        }

        let result: Result<ImportResult, Error> = await Task.detached(priority: .userInitiated) {
            do {
                return .success(try Self.copyPackages(
                    urls: urls,
                    into: folderID,
                    destinations: destinations,
                    existingIDs: existingIDs,
                    existingNames: existingNames,
                    report: report))
            } catch {
                return .failure(error)
            }
        }.value

        switch result {
        case .success(let importResult):
            state.recordings.append(contentsOf: importResult.recordings)
            save()
            importProgress = nil
            if importResult.failures.isEmpty {
                lastReport = "Imported \(importResult.recordings.count) recording\(importResult.recordings.count == 1 ? "" : "s")."
            } else {
                lastReport = "Imported \(importResult.recordings.count) recording\(importResult.recordings.count == 1 ? "" : "s"). \(importResult.failures.count) could not be imported."
                lastError = importResult.failures.joined(separator: "\n")
            }
        case .failure(let error):
            importProgress = nil
            lastError = error.localizedDescription
        }
    }

    private nonisolated static func copyPackages(
        urls: [URL],
        into folderID: UUID?,
        destinations: AppDirectories,
        existingIDs: Set<UUID>,
        existingNames: Set<String>,
        report: @escaping @Sendable (String, Double, Int, Int) -> Void
    ) throws -> ImportResult {
        try destinations.prepare()
        var packageURLs: [URL] = []
        for url in urls {
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else { continue }
            if isDirectory.boolValue {
                if RecordingInspector.looksLikePackage(url) {
                    packageURLs.append(url)
                } else {
                    let children = (try? FileManager.default.contentsOfDirectory(
                        at: url, includingPropertiesForKeys: [.isDirectoryKey],
                        options: [.skipsHiddenFiles])) ?? []
                    for child in children {
                        var childIsDirectory: ObjCBool = false
                        if FileManager.default.fileExists(atPath: child.path, isDirectory: &childIsDirectory),
                           childIsDirectory.boolValue,
                           RecordingInspector.looksLikePackage(child) {
                            packageURLs.append(child)
                        }
                    }
                }
            }
        }
        var unique: [URL] = []
        var seen = Set<String>()
        for url in packageURLs where seen.insert(url.standardizedFileURL.path).inserted {
            unique.append(url)
        }
        guard !unique.isEmpty else {
            throw WorldTrainerError.importFailure("No recording packages were found. Select a folder containing manifest.json, events.atrevents, and capture.mov, or a folder of such packages.")
        }

        var usedIDs = existingIDs
        var usedNames = existingNames
        var imported: [LibraryRecording] = []
        var failures: [String] = []

        for (index, packageURL) in unique.enumerated() {
            report("Inspecting \(packageURL.lastPathComponent)…", Double(index) / Double(unique.count), index, unique.count)
            let inspection = try? awaitInspect(packageURL)
            let preferredID = inspection?.manifest?.id
            var catalogID = preferredID.flatMap { usedIDs.contains($0) ? nil : $0 } ?? UUID()
            while usedIDs.contains(catalogID) { catalogID = UUID() }
            usedIDs.insert(catalogID)
            var directoryName = catalogID.uuidString
            while usedNames.contains(directoryName) {
                catalogID = UUID()
                usedIDs.insert(catalogID)
                directoryName = catalogID.uuidString
            }
            usedNames.insert(directoryName)
            let destination = destinations.recordingsDirectory.appendingPathComponent(directoryName, isDirectory: true)
            do {
                try FileManager.default.copyItem(at: packageURL, to: destination)
            } catch {
                failures.append("\(packageURL.lastPathComponent): \(error.localizedDescription)")
                continue
            }
            let inspected = inspection ?? RecordingPackageInspection(
                manifest: nil,
                issues: [.error("unreadable", "The package could not be inspected.")],
                eventSummary: nil,
                suggestedName: packageURL.lastPathComponent)
            try? generateThumbnailIfNeeded(packageDirectory: destination, inspection: inspected)
            let recording = LibraryRecording(
                id: catalogID,
                folderID: folderID,
                directoryName: directoryName,
                displayName: inspected.suggestedName,
                importedAt: Date(),
                manifest: inspected.manifest,
                issues: inspected.issues,
                eventSummary: inspected.eventSummary,
                packageBytes: inspected.packageBytes,
                videoWidth: inspected.videoWidth,
                videoHeight: inspected.videoHeight,
                videoDuration: inspected.videoDuration,
                sourcePackageName: packageURL.lastPathComponent)
            imported.append(recording)
            report("Imported \(inspected.suggestedName)", Double(index + 1) / Double(unique.count), index + 1, unique.count)
        }
        return ImportResult(recordings: imported, failures: failures, skipped: unique.count - imported.count)
    }

    /// Bridges the async inspector into the synchronous import copy loop.
    private nonisolated static func awaitInspect(_ url: URL) throws -> RecordingPackageInspection {
        var result: RecordingPackageInspection?
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            result = await RecordingInspector.inspect(packageDirectory: url)
            semaphore.signal()
        }
        semaphore.wait()
        guard let result else {
            throw WorldTrainerError.importFailure("The package at \(url.lastPathComponent) could not be inspected.")
        }
        return result
    }

    private nonisolated static func generateThumbnailIfNeeded(
        packageDirectory: URL,
        inspection: RecordingPackageInspection
    ) throws {
        guard let manifest = inspection.manifest else { return }
        if let name = manifest.thumbnailFile,
           FileManager.default.fileExists(atPath: packageDirectory.appendingPathComponent(name).path) {
            return
        }
        let videoURL = packageDirectory.appendingPathComponent(manifest.videoFile)
        guard FileManager.default.fileExists(atPath: videoURL.path) else { return }
        var image: CGImage?
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .utility) {
            image = try? await VideoFrames.image(
                url: videoURL,
                at: min(manifest.duration * 0.1, 1.0),
                maximumSize: CGSize(width: 480, height: 480))
            semaphore.signal()
        }
        semaphore.wait()
        guard let image else { return }
        let representation = NSBitmapImageRep(cgImage: image)
        guard let data = representation.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) else { return }
        let fileName = manifest.thumbnailFile ?? "thumbnail.jpg"
        try? data.write(to: packageDirectory.appendingPathComponent(fileName), options: .atomic)
        // Keep the catalog thumbnail path in sync even when the manifest did
        // not name one.
        if manifest.thumbnailFile == nil {
            var updated = manifest
            updated.thumbnailFile = fileName
            if let encoded = try? JSONEncoder.iso8601.encode(updated) {
                try? encoded.write(to: packageDirectory.appendingPathComponent("manifest.json"), options: .atomic)
            }
        }
    }
}

private extension JSONEncoder {
    static var iso8601: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}
