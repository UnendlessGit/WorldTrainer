import SwiftUI

enum ModelPreset: String, CaseIterable, Identifiable {
    case quick
    case balanced
    case detailed

    var id: String { rawValue }

    var title: String {
        switch self {
        case .quick: return "Fast iteration"
        case .balanced: return "Balanced (recommended)"
        case .detailed: return "High detail"
        }
    }

    var detail: String {
        switch self {
        case .quick: return "128 × 80 at 20 FPS, smaller network. Trains fastest, generates the quickest frames, coarsest picture."
        case .balanced: return "160 × 104 at 20 FPS. The default balance for 15–30 generated FPS on Apple Silicon."
        case .detailed: return "192 × 120 at 20 FPS, wider network. Sharpest output with the slowest training and generation."
        }
    }

    var configuration: WorldModelConfiguration {
        switch self {
        case .quick:
            var configuration = WorldModelConfiguration()
            configuration.resolution = .compact
            configuration.baseChannels = 24
            configuration.recurrentChannels = 64
            configuration.actionEmbeddingChannels = 16
            configuration.sequenceLength = 10
            configuration.learningRate = 0.0006
            configuration.epochs = 8
            return configuration
        case .balanced:
            return WorldModelConfiguration()
        case .detailed:
            var configuration = WorldModelConfiguration()
            configuration.resolution = .detailed
            configuration.baseChannels = 40
            configuration.recurrentChannels = 128
            configuration.actionEmbeddingChannels = 32
            configuration.sequenceLength = 14
            configuration.learningRate = 0.0003
            configuration.epochs = 12
            return configuration
        }
    }
}

struct ModelsView: View {
    @EnvironmentObject private var store: AppStore
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var models: ModelsStore

    @State private var showingCreate = false
    @State private var deleteTarget: WorldModelRecord?

    var body: some View {
        NavigationSplitView {
            List(selection: $store.modelsSelection) {
                if models.models.isEmpty {
                    Text("No models yet")
                        .foregroundStyle(.secondary)
                        .font(.callout)
                }
                ForEach(models.models) { model in
                    ModelRow(model: model, checkpointCount: models.checkpointCount(for: model.id))
                        .tag(model.id)
                        .contextMenu {
                            Button("Reveal in Finder") { models.reveal(model) }
                            Divider()
                            Button("Delete…", role: .destructive) { deleteTarget = model }
                        }
                }
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(min: 210, ideal: 250)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { showingCreate = true } label: {
                        Label("New Model", systemImage: "plus")
                    }
                    .help("Create a world model and choose what it trains on")
                }
            }
        } detail: {
            if let id = store.modelsSelection, models.model(id: id) != nil {
                ModelEditorView(modelID: id, onDelete: {
                    if let model = models.model(id: id) { deleteTarget = model }
                })
                .id(id)
            } else {
                EmptyStateView(
                    systemImage: "brain",
                    title: "No model selected",
                    message: "A world model learns to continue a game from recorded frames and controls. Create one, pick the recordings it learns from, then train it.",
                    actionTitle: "Create a World Model",
                    action: { showingCreate = true })
            }
        }
        .sheet(isPresented: $showingCreate) {
            CreateModelSheet { name, preset, recordingIDs in
                let model = models.create(name: name, configuration: preset.configuration, recordingIDs: recordingIDs)
                store.modelsSelection = model.id
                store.trainingModelID = model.id
                store.runModelID = model.id
                showingCreate = false
            } onCancel: {
                showingCreate = false
            }
        }
        .confirmationDialog(
            "Delete \(deleteTarget?.name ?? "model")?",
            isPresented: Binding(get: { deleteTarget != nil }, set: { if !$0 { deleteTarget = nil } }),
            presenting: deleteTarget
        ) { model in
            Button("Delete Model and Checkpoints", role: .destructive) {
                models.delete(model, removeFiles: true)
                if store.modelsSelection == model.id { store.modelsSelection = nil }
                if store.trainingModelID == model.id { store.trainingModelID = nil }
                if store.runModelID == model.id { store.runModelID = nil }
                deleteTarget = nil
            }
            Button("Remove From List Only") {
                models.delete(model, removeFiles: false)
                if store.modelsSelection == model.id { store.modelsSelection = nil }
                deleteTarget = nil
            }
            Button("Cancel", role: .cancel) { deleteTarget = nil }
        } message: { model in
            Text("Deleting removes the saved weights and training caches for \(model.name). Recordings in the library are never touched.")
        }
    }
}

struct ModelRow: View {
    let model: WorldModelRecord
    let checkpointCount: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Text(model.name)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                if model.isTrained {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.caption2)
                        .foregroundStyle(.green)
                }
            }
            HStack(spacing: 6) {
                Text(model.configuration.resolution.rawValue)
                Text("·")
                Text("\(model.recordingIDs.count) recordings")
                if checkpointCount > 0 {
                    Text("·")
                    Text("\(checkpointCount) ckpt")
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

struct CreateModelSheet: View {
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var models: ModelsStore

    let onCreate: (String, ModelPreset, [UUID]) -> Void
    let onCancel: () -> Void

    @State private var name = ""
    @State private var preset: ModelPreset = .balanced
    @State private var selection: Set<UUID> = []
    @State private var useAllUsable = true

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("New World Model")
                .font(.title3.weight(.semibold))

            TextField("Name", text: $name)
                .textFieldStyle(.roundedBorder)

            VStack(alignment: .leading, spacing: 6) {
                Text("Starting configuration").font(.subheadline.weight(.semibold))
                Picker("", selection: $preset) {
                    ForEach(ModelPreset.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()
                Text(preset.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            VStack(alignment: .leading, spacing: 6) {
                Toggle("Train on every usable recording", isOn: $useAllUsable)
                if !useAllUsable {
                    DatasetPicker(selection: $selection)
                        .frame(height: 180)
                } else {
                    Text("\(library.usableRecordings.count) usable recording\(library.usableRecordings.count == 1 ? "" : "s") will be selected. You can change this later.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Create") {
                    let ids = useAllUsable ? library.usableRecordings.map(\.id) : Array(selection)
                    onCreate(name.trimmingCharacters(in: .whitespacesAndNewlines), preset, ids)
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .disabled(library.usableRecordings.isEmpty)
            }
            if library.usableRecordings.isEmpty {
                Text("Import at least one usable recording in the Library tab first.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding(20)
        .frame(width: 520)
    }
}

/// Recording selection grouped by library folder.
struct DatasetPicker: View {
    @EnvironmentObject private var library: LibraryStore
    @Binding var selection: Set<UUID>

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(library.folders) { folder in
                    let recordings = library.recordings(in: folder.id)
                    if !recordings.isEmpty {
                        group(title: folder.name, recordings: recordings)
                    }
                }
                let unfiled = library.recordings(in: nil)
                if !unfiled.isEmpty {
                    group(title: "Unfiled", recordings: unfiled)
                }
                if library.recordings.isEmpty {
                    Text("The library is empty.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(8)
        }
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func group(title: String, recordings: [LibraryRecording]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(recordings) { recording in
                Toggle(isOn: binding(for: recording.id)) {
                    HStack(spacing: 6) {
                        Text(recording.displayName)
                            .font(.callout)
                            .lineLimit(1)
                        if recording.usability == .unusable {
                            Badge(text: "Unusable", tint: .red)
                        }
                    }
                }
                .toggleStyle(.checkbox)
                .disabled(recording.usability == .unusable)
                .help(recording.usability == .unusable
                      ? recording.errorMessages.joined(separator: " ")
                      : "\(recording.eventCount) events · \(Format.duration(recording.duration))")
            }
        }
    }

    private func binding(for id: UUID) -> Binding<Bool> {
        Binding(
            get: { selection.contains(id) },
            set: { value in
                if value { selection.insert(id) } else { selection.remove(id) }
            })
    }
}

struct ModelEditorView: View {
    @EnvironmentObject private var store: AppStore
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var models: ModelsStore

    let modelID: UUID
    let onDelete: () -> Void

    @State private var draft: WorldModelRecord?
    @State private var deleteCheckpointTarget: TrainingCheckpoint?
    @State private var showAdvanced = true

    private var storedModel: WorldModelRecord? { models.model(id: modelID) }

    var body: some View {
        Group {
            if let draft, let storedModel {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        header(draft, stored: storedModel)
                        datasets(draft)
                        configuration(draft)
                        checkpoints(storedModel)
                        notes(draft)
                    }
                    .padding(18)
                }
            } else {
                EmptyStateView(systemImage: "brain", title: "Model unavailable", message: "This model is no longer in the library index.")
            }
        }
        .navigationTitle(storedModel?.name ?? "Model")
        .onAppear { syncDraft() }
        .onChange(of: models.models) { _, _ in syncDraft() }
        .confirmationDialog(
            "Delete checkpoint?",
            isPresented: Binding(get: { deleteCheckpointTarget != nil }, set: { if !$0 { deleteCheckpointTarget = nil } }),
            presenting: deleteCheckpointTarget
        ) { checkpoint in
            Button("Delete Checkpoint", role: .destructive) {
                models.deleteCheckpoint(modelID: modelID, checkpointID: checkpoint.id)
                deleteCheckpointTarget = nil
            }
            Button("Cancel", role: .cancel) { deleteCheckpointTarget = nil }
        } message: { checkpoint in
            Text("\(checkpoint.label) will be removed from disk.")
        }
    }

    private func syncDraft() {
        guard let stored = storedModel else { return }
        if draft == nil || draft != stored { draft = stored }
    }

    private func save(_ transform: (inout WorldModelRecord) -> Void) {
        guard var updated = draft else { return }
        transform(&updated)
        draft = updated
        models.update(updated)
    }

    // MARK: - Header

    private func header(_ model: WorldModelRecord, stored: WorldModelRecord) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 14) {
                VStack(alignment: .leading, spacing: 6) {
                    TextField("Model name", text: Binding(
                        get: { model.name },
                        set: { value in save { $0.name = value } }))
                        .textFieldStyle(.plain)
                        .font(.title2.weight(.semibold))
                        .frame(maxWidth: 420)
                    HStack(spacing: 6) {
                        Badge(
                            text: stored.isTrained ? "Trained" : "Untrained",
                            tint: stored.isTrained ? .green : .orange,
                            systemImage: stored.isTrained ? "checkmark.seal.fill" : "hourglass")
                        Badge(text: model.configuration.resolution.rawValue, tint: .secondary)
                        Badge(text: "\(Format.integer(model.configuration.estimatedParameterCount())) parameters", tint: .secondary)
                        Badge(text: "\(model.recordingIDs.count) recordings", tint: .secondary)
                        if let trainedAt = stored.trainedAt {
                            Badge(text: "trained \(trainedAt.formatted(date: .abbreviated, time: .shortened))", tint: .secondary)
                        }
                    }
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    Button {
                        store.trainingModelID = model.id
                        store.selectTab(.training)
                    } label: {
                        Label("Train This Model", systemImage: "chart.line.uptrend.xyaxis")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.recordingIDs.isEmpty)
                    HStack(spacing: 8) {
                        Button { models.reveal(model) } label: {
                            Label("Reveal", systemImage: "folder")
                        }
                        Button(role: .destructive) { onDelete() } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                }
            }
            if let error = models.lastError {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(.red)
            }
        }
    }

    // MARK: - Datasets

    private func datasets(_ model: WorldModelRecord) -> some View {
        let sources = models.sources(for: model, library: library)
        let capabilities = models.capabilities(for: model, library: library)
        let usable = sources.filter { $0.manifest.isStructurallyValid }
        let frames = usable.reduce(0) { total, source in
            let start = source.manifest.effectiveTrimStart
            let end = max(start, source.manifest.effectiveTrimEnd)
            return total + Int((end - start) * model.configuration.targetFPS)
        }
        return SectionCard(
            title: "Training data",
            subtitle: "Only recordings selected here are learned. Changing the selection invalidates exact checkpoint resume.",
            systemImage: "square.stack.3d.up"
        ) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Button("Select All Usable") {
                        save { $0.recordingIDs = library.usableRecordings.map(\.id) }
                    }
                    Button("Clear") { save { $0.recordingIDs = [] } }
                    Spacer()
                    Text("\(usable.count) usable · about \(Format.integer(frames)) frames at \(Int(model.configuration.targetFPS)) FPS")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                DatasetPicker(selection: Binding(
                    get: { Set(model.recordingIDs) },
                    set: { newValue in save { $0.recordingIDs = Array(newValue) } }))
                    .frame(height: 200)
                if usable.count < 2 {
                    Label("Add at least two recordings so validation can hold out separate footage. With one recording, the tail is held out instead.",
                          systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Divider()
                Text("Control channels available to Run")
                    .font(.subheadline.weight(.semibold))
                CapabilitySummary(
                    capabilities: capabilities,
                    emptyMessage: "The selected recordings contain no input events, so the model can only learn video, not controls.")
            }
        }
    }

    // MARK: - Configuration

    private func configuration(_ model: WorldModelRecord) -> some View {
        let issues = model.configuration.validationIssues()
        return SectionCard(
            title: "Architecture and training",
            subtitle: "Every field here changes what is trained. Architecture changes require a fresh run; training settings are stored with each checkpoint.",
            systemImage: "slider.horizontal.3"
        ) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Menu("Apply Preset") {
                        ForEach(ModelPreset.allCases) { preset in
                            Button(preset.title) {
                                save { model in
                                    let recordings = model.recordingIDs
                                    model.configuration = preset.configuration
                                    model.recordingIDs = recordings
                                }
                            }
                        }
                    }
                    .frame(width: 140)
                    Toggle("Show advanced settings", isOn: $showAdvanced)
                        .toggleStyle(.switch)
                        .controlSize(.small)
                    Spacer()
                }

                Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 10) {
                    GridRow {
                        Text("Resolution").foregroundStyle(.secondary)
                        Picker("", selection: configBinding(\.resolution)) {
                            ForEach(ModelResolution.allCases) { resolution in
                                Text("\(resolution.label) — \(resolution.detail)").tag(resolution)
                            }
                        }
                        .labelsHidden()
                        .frame(maxWidth: 420)
                    }
                    GridRow {
                        Text("Target FPS").foregroundStyle(.secondary)
                        numberField(configBinding(\.targetFPS), format: .number.precision(.fractionLength(0)), width: 70)
                    }
                    GridRow {
                        Text("Sequence length").foregroundStyle(.secondary)
                        HStack(spacing: 8) {
                            Stepper(value: intBinding(\.sequenceLength), in: 4...32) {
                                Text("\(model.configuration.sequenceLength) frames")
                            }
                            Text("Longer windows teach motion over more time but need more memory.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    GridRow {
                        Text("Batch size").foregroundStyle(.secondary)
                        Stepper(value: intBinding(\.batchSize), in: 1...8) {
                            Text("\(model.configuration.batchSize)")
                        }
                        .frame(width: 130)
                    }
                    GridRow {
                        Text("Epochs").foregroundStyle(.secondary)
                        Stepper(value: intBinding(\.epochs), in: 1...500) {
                            Text("\(model.configuration.epochs)")
                        }
                        .frame(width: 130)
                    }
                    if showAdvanced {
                        GridRow {
                            Text("Base channels").foregroundStyle(.secondary)
                            Stepper(value: intBinding(\.baseChannels), in: 16...64) {
                                Text("\(model.configuration.baseChannels)")
                            }
                            .frame(width: 130)
                        }
                        GridRow {
                            Text("Recurrent channels").foregroundStyle(.secondary)
                            Stepper(value: intBinding(\.recurrentChannels), in: 32...192) {
                                Text("\(model.configuration.recurrentChannels)")
                            }
                            .frame(width: 130)
                        }
                        GridRow {
                            Text("Action embedding").foregroundStyle(.secondary)
                            Stepper(value: intBinding(\.actionEmbeddingChannels), in: 8...96) {
                                Text("\(model.configuration.actionEmbeddingChannels)")
                            }
                            .frame(width: 130)
                        }
                        GridRow {
                            Text("Learning rate").foregroundStyle(.secondary)
                            numberField(configBinding(\.learningRate), format: .number.precision(.fractionLength(5)), width: 90)
                        }
                        GridRow {
                            Text("Weight decay").foregroundStyle(.secondary)
                            numberField(configBinding(\.weightDecay), format: .number.precision(.fractionLength(5)), width: 90)
                        }
                        GridRow {
                            Text("Gradient clip").foregroundStyle(.secondary)
                            numberField(configBinding(\.gradientClip), format: .number.precision(.fractionLength(1)), width: 70)
                        }
                        GridRow {
                            Text("Validation fraction").foregroundStyle(.secondary)
                            numberField(configBinding(\.validationFraction), format: .number.precision(.fractionLength(2)), width: 70)
                        }
                        GridRow {
                            Text("Temporal loss weight").foregroundStyle(.secondary)
                            numberField(configBinding(\.temporalConsistencyWeight), format: .number.precision(.fractionLength(2)), width: 70)
                        }
                        GridRow {
                            Text("Edge loss weight").foregroundStyle(.secondary)
                            numberField(configBinding(\.edgeLossWeight), format: .number.precision(.fractionLength(2)), width: 70)
                        }
                        GridRow {
                            Text("Random seed").foregroundStyle(.secondary)
                            HStack(spacing: 8) {
                                numberField(configBinding(\.seed), format: .number.precision(.fractionLength(0)), width: 90)
                                Button("Randomise") { save { $0.configuration.seed = UInt64.random(in: 1...9_999_999) } }
                            }
                        }
                    }
                }

                if !issues.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(issues, id: \.self) { issue in
                            Label(issue, systemImage: "exclamationmark.triangle.fill")
                                .font(.callout)
                                .foregroundStyle(.orange)
                        }
                    }
                }

                HStack(spacing: 14) {
                    Label("\(Format.integer(model.configuration.estimatedParameterCount())) parameters", systemImage: "number")
                    Label("\(model.configuration.width) × \(model.configuration.height) output", systemImage: "rectangle.on.rectangle")
                    Label("\(ActionLayout.count) control values", systemImage: "gamecontroller")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }

    private func configBinding<Value>(_ keyPath: WritableKeyPath<WorldModelConfiguration, Value>) -> Binding<Value> {
        Binding(
            get: { (draft?.configuration ?? WorldModelConfiguration())[keyPath: keyPath] },
            set: { value in save { $0.configuration[keyPath: keyPath] = value } })
    }

    private func intBinding(_ keyPath: WritableKeyPath<WorldModelConfiguration, Int>) -> Binding<Int> {
        configBinding(keyPath)
    }

    private func numberField(_ binding: Binding<Double>, format: FloatingPointFormatStyle<Double>, width: CGFloat) -> some View {
        TextField("", value: binding, format: format)
            .textFieldStyle(.roundedBorder)
            .frame(width: width)
    }

    private func numberField(_ binding: Binding<UInt64>, format: IntegerFormatStyle<UInt64>, width: CGFloat) -> some View {
        TextField("", value: binding, format: format)
            .textFieldStyle(.roundedBorder)
            .frame(width: width)
    }

    // MARK: - Checkpoints

    private func checkpoints(_ model: WorldModelRecord) -> some View {
        let list = models.checkpoints(for: model.id)
        return SectionCard(
            title: "Checkpoints",
            subtitle: "Every checkpoint holds weights, optimizer state and the exact position in the dataset.",
            systemImage: "clock.arrow.circlepath"
        ) {
            VStack(alignment: .leading, spacing: 8) {
                if list.isEmpty {
                    Text("No checkpoints yet. Training writes one at every epoch boundary, every 200 steps, and whenever you pause or stop.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(list) { checkpoint in
                        HStack(spacing: 10) {
                            Image(systemName: checkpoint.kind == .best ? "star.fill" : "circle.fill")
                                .font(.caption2)
                                .foregroundStyle(checkpoint.kind == .best ? .yellow : .secondary)
                            VStack(alignment: .leading, spacing: 2) {
                                HStack(spacing: 6) {
                                    Text(checkpoint.label).font(.callout.weight(.medium))
                                    Badge(text: checkpoint.kind.rawValue, tint: .secondary)
                                    if model.trainedCheckpointID == checkpoint.id {
                                        Badge(text: "Used for Run", tint: .green, systemImage: "play.fill")
                                    }
                                }
                                Text(detailLine(checkpoint))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Use for Run") {
                                models.markTrained(modelID: model.id, checkpoint: checkpoint)
                            }
                            .disabled(model.trainedCheckpointID == checkpoint.id)
                            Button {
                                models.revealCheckpoint(modelID: model.id, checkpointID: checkpoint.id)
                            } label: {
                                Image(systemName: "folder")
                            }
                            .help("Reveal checkpoint in Finder")
                            Button(role: .destructive) {
                                deleteCheckpointTarget = checkpoint
                            } label: {
                                Image(systemName: "trash")
                            }
                            .help("Delete this checkpoint")
                        }
                        .padding(.vertical, 3)
                        if checkpoint.id != list.last?.id { Divider() }
                    }
                }
            }
        }
    }

    private func detailLine(_ checkpoint: TrainingCheckpoint) -> String {
        var parts: [String] = []
        parts.append(checkpoint.createdAt.formatted(date: .abbreviated, time: .shortened))
        if let training = checkpoint.trainingLoss { parts.append("train \(Format.decimal(Double(training)))") }
        if let validation = checkpoint.validationLoss { parts.append("val \(Format.decimal(Double(validation)))") }
        parts.append("\(Format.integer(checkpoint.datasetRecordingIDs.count)) recordings")
        parts.append("\(Format.integer(checkpoint.parameterCount)) params")
        return parts.joined(separator: " · ")
    }

    // MARK: - Notes

    private func notes(_ model: WorldModelRecord) -> some View {
        SectionCard(title: "Notes", systemImage: "text.alignleft") {
            TextEditor(text: Binding(
                get: { model.notes },
                set: { value in save { $0.notes = value } }))
                .font(.body)
                .frame(minHeight: 80)
                .scrollContentBackground(.hidden)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
        }
    }
}
