import AVFoundation
import AppKit
import SwiftUI

enum LibraryScope: Hashable {
    case all
    case unfiled
    case unusable
    case folder(UUID)
}

struct LibraryView: View {
    @EnvironmentObject private var store: AppStore
    @EnvironmentObject private var library: LibraryStore

    @State private var scope: LibraryScope = .all
    @State private var search = ""
    @State private var showingImport = false
    @State private var showingNewFolder = false
    @State private var newFolderName = ""
    @State private var renameFolderTarget: LibraryFolder?
    @State private var renameFolderName = ""
    @State private var renameRecordingTarget: LibraryRecording?
    @State private var renameRecordingName = ""
    @State private var deleteTarget: LibraryRecording?

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 190, ideal: 215, max: 280)
        } content: {
            recordingList
                .navigationSplitViewColumnWidth(min: 300, ideal: 380)
        } detail: {
            detail
        }
        .fileImporter(
            isPresented: $showingImport,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls):
                let folderID: UUID? = {
                    if case .folder(let id) = scope { return id }
                    return nil
                }()
                Task { await library.importPackages(from: urls, into: folderID) }
            case .failure(let error):
                library.lastError = error.localizedDescription
            }
        }
        .sheet(isPresented: $showingNewFolder) {
            FolderNameSheet(
                title: "New Folder",
                prompt: "Name",
                initial: "",
                confirmTitle: "Create"
            ) { name in
                library.addFolder(named: name)
                showingNewFolder = false
            } onCancel: {
                showingNewFolder = false
            }
        }
        .sheet(item: $renameFolderTarget) { folder in
            FolderNameSheet(
                title: "Rename Folder",
                prompt: "Name",
                initial: folder.name,
                confirmTitle: "Rename"
            ) { name in
                library.renameFolder(folder, to: name)
                renameFolderTarget = nil
            } onCancel: {
                renameFolderTarget = nil
            }
        }
        .sheet(item: $renameRecordingTarget) { recording in
            FolderNameSheet(
                title: "Rename Recording",
                prompt: "Name",
                initial: recording.displayName,
                confirmTitle: "Rename"
            ) { name in
                library.rename(recording, to: name)
                renameRecordingTarget = nil
            } onCancel: {
                renameRecordingTarget = nil
            }
        }
        .confirmationDialog(
            "Remove \(deleteTarget?.displayName ?? "recording")?",
            isPresented: Binding(get: { deleteTarget != nil }, set: { if !$0 { deleteTarget = nil } }),
            presenting: deleteTarget
        ) { recording in
            Button("Delete Recording and Files", role: .destructive) {
                library.delete(recording, removeFiles: true)
                if store.libraryRecordingSelection == recording.id { store.libraryRecordingSelection = nil }
                deleteTarget = nil
            }
            Button("Remove From Library Only") {
                library.delete(recording, removeFiles: false)
                if store.libraryRecordingSelection == recording.id { store.libraryRecordingSelection = nil }
                deleteTarget = nil
            }
            Button("Cancel", role: .cancel) { deleteTarget = nil }
        } message: { recording in
            Text("Deleting files frees \(Format.bytes(recording.packageBytes)). Models that already trained on it keep their checkpoints.")
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(selection: $scope) {
            Section("Library") {
                Label {
                    countLabel("All Recordings", library.recordings.count)
                } icon: {
                    Image(systemName: "square.stack.3d.up")
                }
                .tag(LibraryScope.all)

                Label {
                    countLabel("Unfiled", library.recordings(in: nil).count)
                } icon: {
                    Image(systemName: "tray")
                }
                .tag(LibraryScope.unfiled)

                Label {
                    countLabel("Unusable", unusableCount)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .tag(LibraryScope.unusable)
            }
            Section("Folders") {
                ForEach(library.folders) { folder in
                    Label {
                        countLabel(folder.name, library.recordings(in: folder.id).count)
                    } icon: {
                        Image(systemName: "folder")
                    }
                    .tag(LibraryScope.folder(folder.id))
                    .contextMenu {
                        Button("Rename…") {
                            renameFolderName = folder.name
                            renameFolderTarget = folder
                        }
                        Button("Delete Folder", role: .destructive) {
                            library.deleteFolder(folder)
                            if scope == .folder(folder.id) { scope = .all }
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showingNewFolder = true
                } label: {
                    Label("New Folder", systemImage: "folder.badge.plus")
                }
                .help("Create a folder for organising recordings")
            }
        }
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading, spacing: 6) {
                if let report = library.lastReport {
                    Text(report)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Button {
                    showingImport = true
                } label: {
                    Label("Import Recordings…", systemImage: "square.and.arrow.down")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(library.importProgress != nil)
                Text(importHint)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(10)
            .background(.bar)
        }
    }

    private var importHint: String {
        if case .folder(let id) = scope, let folder = library.folders.first(where: { $0.id == id }) {
            return "New recordings are added to \(folder.name)."
        }
        return "Select a folder to import into it, or import to Unfiled."
    }

    private func countLabel(_ title: String, _ count: Int) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text("\(count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    private var unusableCount: Int {
        library.recordings.filter { $0.usability == .unusable }.count
    }

    // MARK: - List

    private var filteredRecordings: [LibraryRecording] {
        let scoped: [LibraryRecording]
        switch scope {
        case .all: scoped = library.recordings
        case .unfiled: scoped = library.recordings.filter { $0.folderID == nil }
        case .unusable: scoped = library.recordings.filter { $0.usability == .unusable }
        case .folder(let id): scoped = library.recordings.filter { $0.folderID == id }
        }
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let searched = query.isEmpty ? scoped : scoped.filter {
            $0.displayName.lowercased().contains(query)
                || ($0.sourcePackageName?.lowercased().contains(query) ?? false)
        }
        return searched.sorted { $0.importedAt > $1.importedAt }
    }

    private var recordingList: some View {
        VStack(spacing: 0) {
            if library.importProgress != nil {
                ImportProgressBar(progress: library.importProgress)
            }
            if filteredRecordings.isEmpty {
                EmptyStateView(
                    systemImage: "tray",
                    title: "No recordings here",
                    message: "Import recording packages made by the recorder. WorldTrainer reads them and never records on its own.",
                    actionTitle: "Import Recordings…",
                    action: { showingImport = true })
            } else {
                List(selection: $store.libraryRecordingSelection) {
                    ForEach(filteredRecordings) { recording in
                        RecordingRow(
                            recording: recording,
                            thumbnailURL: library.thumbnailURL(for: recording),
                            showsFolder: scope != .folder(recording.folderID ?? UUID()),
                            folderName: folderName(for: recording))
                            .tag(recording.id)
                            .contextMenu {
                                Button("Rename…") {
                                    renameRecordingName = recording.displayName
                                    renameRecordingTarget = recording
                                }
                                Menu("Move to Folder") {
                                    Button("Unfiled") { library.move(recording, to: nil) }
                                    Divider()
                                    ForEach(library.folders) { folder in
                                        Button(folder.name) { library.move(recording, to: folder.id) }
                                    }
                                }
                                Button("Reveal in Finder") { library.reveal(recording) }
                                Divider()
                                Button("Delete…", role: .destructive) { deleteTarget = recording }
                            }
                    }
                }
                .listStyle(.inset)
            }
        }
        .searchable(text: $search, placement: .toolbar, prompt: "Search recordings")
        .navigationTitle(navigationTitle)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    library.refresh()
                } label: {
                    Label("Recheck Packages", systemImage: "arrow.clockwise")
                }
                .help("Re-read package files and report anything that changed on disk")
            }
        }
    }

    private var navigationTitle: String {
        switch scope {
        case .all: return "All Recordings"
        case .unfiled: return "Unfiled"
        case .unusable: return "Unusable Recordings"
        case .folder(let id): return library.folders.first { $0.id == id }?.name ?? "Folder"
        }
    }

    private func folderName(for recording: LibraryRecording) -> String? {
        guard let id = recording.folderID else { return nil }
        return library.folders.first { $0.id == id }?.name
    }

    // MARK: - Detail

    @ViewBuilder
    private var detail: some View {
        if let id = store.libraryRecordingSelection, let recording = library.recording(id: id) {
            RecordingDetailView(
                recording: recording,
                directory: library.directory(for: recording),
                thumbnailURL: library.thumbnailURL(for: recording),
                folderName: folderName(for: recording),
                onRename: {
                    renameRecordingName = recording.displayName
                    renameRecordingTarget = recording
                },
                onDelete: { deleteTarget = recording },
                onReveal: { library.reveal(recording) },
                usedByModelCount: modelUsageCount(for: recording.id))
        } else {
            EmptyStateView(
                systemImage: "square.stack.3d.up",
                title: "Select a recording",
                message: "Choose a recording to preview it, inspect its metadata and see which controls it can teach.")
        }
    }

    private func modelUsageCount(for recordingID: UUID) -> Int {
        store.models.models.filter { $0.recordingIDs.contains(recordingID) }.count
    }
}

struct ImportProgressBar: View {
    let progress: ImportProgress?

    var body: some View {
        if let progress {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(progress.message)
                        .font(.caption)
                        .lineLimit(1)
                    Spacer()
                    if progress.total > 0 {
                        Text("\(progress.imported)/\(progress.total)")
                            .font(.caption.monospacedDigit())
                    }
                }
                ProgressView(value: progress.fraction)
                    .progressViewStyle(.linear)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.bar)
        }
    }
}

struct FolderNameSheet: View {
    let title: String
    let prompt: String
    let initial: String
    let confirmTitle: String
    let onConfirm: (String) -> Void
    let onCancel: () -> Void

    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title).font(.headline)
            TextField(prompt, text: $text)
                .textFieldStyle(.roundedBorder)
                .focused($focused)
                .onSubmit { confirm() }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button(confirmTitle) { confirm() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 340)
        .onAppear {
            text = initial
            focused = true
        }
    }

    private func confirm() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        onConfirm(trimmed)
    }
}

struct RecordingDetailView: View {
    let recording: LibraryRecording
    let directory: URL
    let thumbnailURL: URL?
    let folderName: String?
    let onRename: () -> Void
    let onDelete: () -> Void
    let onReveal: () -> Void
    let usedByModelCount: Int

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                if recording.usability == .unusable {
                    unusableBanner
                }
                RecordingPreviewPane(recording: recording, directory: directory)
                SectionCard(title: "Controls this recording can teach", systemImage: "gamecontroller") {
                    CapabilitySummary(capabilities: recording.capabilities)
                }
                SectionCard(title: "Package details", systemImage: "shippingbox") {
                    VStack(alignment: .leading, spacing: 6) {
                        KeyValueRow(key: "Imported", value: recording.importedAt.formatted(date: .abbreviated, time: .shortened))
                        KeyValueRow(key: "Source package", value: recording.sourcePackageName ?? recording.directoryName)
                        KeyValueRow(key: "Folder", value: folderName ?? "Unfiled")
                        KeyValueRow(key: "Stored at", value: directory.path, monospaced: true)
                        KeyValueRow(key: "Package size", value: Format.bytes(recording.packageBytes))
                        KeyValueRow(key: "Used by models", value: "\(usedByModelCount)")
                        Divider().padding(.vertical, 2)
                        KeyValueRow(key: "Recording ID", value: recording.id.uuidString, monospaced: true)
                        if let manifest = recording.manifest {
                            KeyValueRow(key: "Schema version", value: "\(manifest.schemaVersion)")
                            KeyValueRow(key: "Recorded", value: manifest.createdAt.formatted(date: .abbreviated, time: .standard))
                            KeyValueRow(key: "Duration", value: Format.duration(manifest.duration))
                            KeyValueRow(key: "Trim range", value: trimDescription(manifest))
                            KeyValueRow(key: "Delivered FPS", value: Format.rate(manifest.deliveredFPS))
                            KeyValueRow(key: "Requested FPS", value: Format.rate(manifest.requestedFPS))
                            KeyValueRow(key: "Encoded frames", value: Format.integer(Int(manifest.encodedFrameCount)))
                            if let dropped = manifest.droppedFrameCount {
                                KeyValueRow(key: "Dropped frames", value: Format.integer(dropped))
                            }
                            KeyValueRow(key: "Events", value: Format.integer(manifest.eventCount))
                            KeyValueRow(key: "Video", value: manifest.videoFile)
                            KeyValueRow(key: "Events file", value: manifest.eventFile)
                            if let capture = manifest.capture {
                                KeyValueRow(key: "Capture", value: captureDescription(capture))
                            }
                            if let rect = manifest.globalRect {
                                KeyValueRow(key: "Captured region",
                                            value: "\(Int(rect.x)), \(Int(rect.y)) · \(Int(rect.width))×\(Int(rect.height))")
                            }
                        }
                        if let summary = recording.eventSummary {
                            KeyValueRow(key: "Event breakdown", value: eventBreakdown(summary))
                            if let first = summary.firstTimestampNanos, let last = summary.lastTimestampNanos, last > first {
                                let seconds = Double(last - first) / 1_000_000_000
                                KeyValueRow(key: "Event span", value: Format.duration(seconds))
                            }
                        }
                    }
                }
                SectionCard(title: "Validation", systemImage: "checklist") {
                    IssueList(issues: recording.issues)
                }
            }
            .padding(18)
        }
        .navigationTitle(recording.displayName)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button { onRename() } label: { Label("Rename", systemImage: "pencil") }
                Button { onReveal() } label: { Label("Reveal", systemImage: "folder") }
                Button(role: .destructive) { onDelete() } label: { Label("Delete", systemImage: "trash") }
            }
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 14) {
            ThumbnailView(url: thumbnailURL, size: 84, cornerRadius: 8)
            VStack(alignment: .leading, spacing: 5) {
                Text(recording.displayName)
                    .font(.title2.weight(.semibold))
                    .textSelection(.enabled)
                HStack(spacing: 6) {
                    Badge(
                        text: recording.usability.label,
                        tint: recording.usability == .usable ? .green : .red,
                        systemImage: recording.usability == .usable ? "checkmark.circle" : "xmark.octagon")
                    if recording.pixelWidth > 0 {
                        Badge(text: "\(recording.pixelWidth)×\(recording.pixelHeight)", tint: .secondary, systemImage: "rectangle.on.rectangle")
                    }
                    if recording.duration > 0 {
                        Badge(text: Format.duration(recording.duration), tint: .secondary, systemImage: "clock")
                    }
                    Badge(text: "\(Format.integer(recording.eventCount)) events", tint: .secondary, systemImage: "keyboard")
                }
                Text(recording.errorMessages.first ?? "Ready to use for training.")
                    .font(.callout)
                    .foregroundStyle(recording.usability == .usable ? Color.secondary : Color.red)
            }
            Spacer()
        }
    }

    private var unusableBanner: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
            VStack(alignment: .leading, spacing: 3) {
                Text("This recording cannot be used for training")
                    .font(.callout.weight(.semibold))
                Text(recording.errorMessages.joined(separator: " "))
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(12)
        .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
    }

    private func trimDescription(_ manifest: RecordingManifest) -> String {
        let start = Format.duration(manifest.effectiveTrimStart)
        let end = Format.duration(manifest.effectiveTrimEnd)
        return "\(start) → \(end)"
    }

    private func captureDescription(_ capture: CaptureSpec) -> String {
        var parts: [String] = []
        if let kind = capture.kind { parts.append(kind) }
        if let display = capture.displayID { parts.append("display \(display)") }
        if capture.showsCursor == true { parts.append("cursor visible") }
        if let visual = capture.recordingVisuals, let width = visual.width, let height = visual.height {
            parts.append("\(width)×\(height)")
        }
        return parts.isEmpty ? "Not specified" : parts.joined(separator: " · ")
    }

    private func eventBreakdown(_ summary: EventSummary) -> String {
        [
            "\(summary.mouseMoveCount) move",
            "\(summary.mouseButtonCount) button",
            "\(summary.scrollCount) scroll",
            "\(summary.keyEventCount) key",
            "\(summary.flagsEventCount) modifier",
        ].joined(separator: " · ")
    }
}

/// Scrubbable frame preview. Frames are decoded on demand from the recording's
/// own video file; nothing here is generated.
struct RecordingPreviewPane: View {
    let recording: LibraryRecording
    let directory: URL

    @State private var position: Double = 0
    @State private var image: CGImage?
    @State private var isLoading = false
    @State private var error: String?

    var body: some View {
        SectionCard(title: "Preview", subtitle: "Original recorded footage", systemImage: "photo") {
            VStack(alignment: .leading, spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8).fill(.black)
                    if let image {
                        Image(decorative: image, scale: 1)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .padding(2)
                    } else if isLoading {
                        ProgressView()
                    } else {
                        VStack(spacing: 6) {
                            Image(systemName: "film").font(.title2).foregroundStyle(.secondary)
                            Text(error ?? "No preview available")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .frame(height: 240)
                .frame(maxWidth: .infinity)

                HStack(spacing: 10) {
                    Text("0:00").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    Slider(value: $position, in: 0...max(0.2, recording.duration))
                    Text(Format.duration(recording.duration)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Text("Showing \(Format.duration(position)) into the recording")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .task(id: Int(position * 4)) {
            await load()
        }
    }

    private func load() async {
        guard let manifest = recording.manifest else {
            error = "The manifest could not be read."
            return
        }
        let url = directory.appendingPathComponent(manifest.videoFile)
        guard FileManager.default.fileExists(atPath: url.path) else {
            error = "The video file is missing."
            return
        }
        isLoading = true
        defer { isLoading = false }
        do {
            let frame = try await VideoFrames.image(
                url: url, at: position, maximumSize: CGSize(width: 900, height: 900))
            image = frame
            error = nil
        } catch {
            self.error = error.localizedDescription
        }
    }
}
