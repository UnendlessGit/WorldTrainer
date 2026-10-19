import AppKit
import MLX
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var store: AppStore
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var models: ModelsStore
    @EnvironmentObject private var settings: SettingsStore

    @State private var showingRootPicker = false
    @State private var storage: [(String, Int64)] = []
    @State private var storageBusy = false
    @State private var diagnostics: [(String, String)] = []
    @State private var message: String?
    @State private var showingResetConfirm = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                appearance
                storageSection
                performance
                behaviour
                diagnosticsSection
                resetSection
            }
            .padding(18)
            .frame(maxWidth: 900, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .navigationTitle("Settings")
        .onAppear {
            refreshStorage()
            refreshDiagnostics()
        }
        .fileImporter(
            isPresented: $showingRootPicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                settings.preferences.dataRootPath = url.path
                store.applyStorageLocationChange()
                message = "Storage location changed to \(url.path). Existing files stay where they are until you move them."
                refreshStorage()
            case .failure(let error):
                message = error.localizedDescription
            }
        }
        .confirmationDialog("Reset all settings?", isPresented: $showingResetConfirm) {
            Button("Reset Settings", role: .destructive) {
                settings.reset()
                store.applyStorageLocationChange()
                message = "Settings reset to their defaults."
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Recordings, models and checkpoints are not deleted.")
        }
    }

    // MARK: - Appearance

    private var appearance: some View {
        SectionCard(title: "Appearance", systemImage: "paintbrush") {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Theme", selection: Binding(
                    get: { settings.preferences.appearance },
                    set: { settings.preferences.appearance = $0 })) {
                    ForEach(AppAppearance.allCases) { option in
                        Text(option.label).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 320)
                Text("The theme applies immediately to every window.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Storage

    private var storageSection: some View {
        SectionCard(
            title: "Storage",
            subtitle: "Where recordings, models, checkpoints and training caches live.",
            systemImage: "externaldrive"
        ) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Text(settings.preferences.dataRootPath)
                        .font(.callout.monospaced())
                        .textSelection(.enabled)
                        .lineLimit(2)
                    Spacer()
                    Button("Choose…") { showingRootPicker = true }
                    Button("Reveal") {
                        NSWorkspace.shared.activateFileViewerSelecting([store.directories.root])
                    }
                }

                Divider()

                HStack(spacing: 10) {
                    Text("On disk")
                        .font(.subheadline.weight(.semibold))
                    if storageBusy { ProgressView().controlSize(.small) }
                    Spacer()
                    Button("Refresh") { refreshStorage() }
                        .controlSize(.small)
                    Button("Clear Training Caches") { clearCaches() }
                        .controlSize(.small)
                        .help("Frees decoded frame and control caches. They are rebuilt automatically the next time you train.")
                }

                if storage.isEmpty {
                    Text("Calculating…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    VStack(alignment: .leading, spacing: 5) {
                        ForEach(storage, id: \.0) { entry in
                            HStack {
                                Text(entry.0)
                                Spacer()
                                Text(Format.bytes(entry.1))
                                    .font(.callout.monospacedDigit())
                                    .foregroundStyle(entry.0 == "Total" ? .primary : .secondary)
                            }
                            .font(entry.0 == "Total" ? .callout.weight(.semibold) : .callout)
                        }
                    }
                }

                KeyValueRow(key: "Library file", value: store.directories.libraryFile.path, monospaced: true)
                KeyValueRow(key: "Models index", value: store.directories.modelsFile.path, monospaced: true)
            }
        }
    }

    // MARK: - Performance

    private var performance: some View {
        SectionCard(
            title: "Performance",
            subtitle: "Applied to the next training or generation session.",
            systemImage: "gauge.with.dots.needle.67percent"
        ) {
            VStack(alignment: .leading, spacing: 14) {
                limitRow(
                    title: "MLX memory limit",
                    detail: "Upper bound for model, activations and optimizer state. Lower it if macOS starts swapping.",
                    value: Binding(
                        get: { settings.preferences.memoryLimitGB },
                        set: { settings.preferences.memoryLimitGB = $0 }),
                    range: 2...48)

                limitRow(
                    title: "MLX cache limit",
                    detail: "Reusable buffer cache. Larger values trade memory for fewer allocations.",
                    value: Binding(
                        get: { settings.preferences.cacheLimitGB },
                        set: { settings.preferences.cacheLimitGB = $0 }),
                    range: 0.25...16)

                limitRow(
                    title: "Dataset cache budget",
                    detail: "Guideline for decoded training caches. Clear them any time from Storage.",
                    value: Binding(
                        get: { settings.preferences.datasetCacheLimitGB },
                        set: { settings.preferences.datasetCacheLimitGB = $0 }),
                    range: 1...200)

                Divider()

                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Default generation rate")
                        Text("Starting target for new Run sessions. The model is not capped by this; it only paces the output.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Slider(value: Binding(
                        get: { settings.preferences.defaultRunFPS },
                        set: { settings.preferences.defaultRunFPS = $0 }),
                           in: 5...120, step: 1)
                        .frame(width: 180)
                    Text("\(Int(settings.preferences.defaultRunFPS)) FPS")
                        .font(.callout.monospacedDigit())
                        .frame(width: 62, alignment: .trailing)
                }

                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Warm-up frames")
                        Text("Recorded frames used only to fill recurrent memory before generating.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Stepper("\(settings.preferences.runWarmupFrames)", value: Binding(
                        get: { settings.preferences.runWarmupFrames },
                        set: { settings.preferences.runWarmupFrames = $0 }),
                            in: 4...120, step: 4)
                        .frame(width: 120)
                }

                Toggle("Show advanced metrics in Training", isOn: Binding(
                    get: { settings.preferences.showAdvancedMetrics },
                    set: { settings.preferences.showAdvancedMetrics = $0 }))

                HStack(spacing: 8) {
                    Button("Reset MLX Cache") {
                        Memory.clearCache()
                        message = "MLX buffer cache cleared. Active memory was \(Format.bytes(Int64(Memory.activeMemory)))."
                        refreshDiagnostics()
                    }
                    Text("Active \(Format.bytes(Int64(Memory.activeMemory))) · peak \(Format.bytes(Int64(Memory.peakMemory)))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func limitRow(
        title: String,
        detail: String,
        value: Binding<Double>,
        range: ClosedRange<Double>
    ) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            Slider(value: value, in: range)
                .frame(width: 170)
            Text(value.wrappedValue < 1
                 ? "\(Format.decimal(value.wrappedValue, places: 2)) GB"
                 : "\(Format.decimal(value.wrappedValue, places: 0)) GB")
                .font(.callout.monospacedDigit())
                .frame(width: 72, alignment: .trailing)
        }
    }

    // MARK: - Behaviour

    private var behaviour: some View {
        SectionCard(title: "Behaviour", systemImage: "switch.2") {
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Confirm before deleting recordings or models", isOn: Binding(
                    get: { settings.preferences.confirmBeforeDelete },
                    set: { settings.preferences.confirmBeforeDelete = $0 }))
                Toggle("Keep only the best checkpoint at epoch boundaries", isOn: Binding(
                    get: { settings.preferences.keepBestCheckpointsOnly },
                    set: { settings.preferences.keepBestCheckpointsOnly = $0 }))
                Text("Checkpoints always include weights, optimizer state and the dataset cursor, so a paused run continues exactly where it stopped.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Diagnostics

    private var diagnosticsSection: some View {
        SectionCard(
            title: "Diagnostics",
            subtitle: "What WorldTrainer is running on, and where its files are.",
            systemImage: "stethoscope"
        ) {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(diagnostics, id: \.0) { entry in
                    KeyValueRow(key: entry.0, value: entry.1, monospaced: entry.0.contains("memory") || entry.0.contains("root"))
                }
                HStack(spacing: 8) {
                    Button("Refresh") { refreshDiagnostics() }
                    Button("Copy Report") {
                        let text = diagnostics.map { "\($0.0): \($0.1)" }.joined(separator: "\n")
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(text, forType: .string)
                        message = "Diagnostics copied to the clipboard."
                    }
                    Spacer()
                    Text("\(library.recordings.count) recordings · \(models.models.count) models")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var resetSection: some View {
        SectionCard(title: "Reset", systemImage: "arrow.counterclockwise") {
            VStack(alignment: .leading, spacing: 8) {
                Button("Reset All Settings") { showingResetConfirm = true }
                if let message {
                    Text(message)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
    }

    // MARK: - Actions

    private func refreshStorage() {
        storageBusy = true
        let directories = store.directories
        Task {
            let result: [(String, Int64)] = await Task.detached(priority: .utility) {
                [
                    ("Recordings", AppDirectories.directoryBytes(directories.recordingsDirectory)),
                    ("Models & checkpoints", AppDirectories.directoryBytes(directories.modelsDirectory)),
                    ("Training caches", AppDirectories.directoryBytes(directories.trainingCacheDirectory)),
                    ("Total", AppDirectories.directoryBytes(directories.root)),
                ]
            }.value
            storage = result
            storageBusy = false
        }
    }

    private func clearCaches() {
        do {
            try settings.clearTrainingCaches()
            message = "Training caches cleared. They will be rebuilt on the next training run."
        } catch {
            message = "Caches could not be cleared: \(error.localizedDescription)"
        }
        refreshStorage()
    }

    private func refreshDiagnostics() {
        diagnostics = store.diagnostics()
    }
}
