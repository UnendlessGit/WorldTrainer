import SwiftUI

struct TrainingView: View {
    @EnvironmentObject private var store: AppStore
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var models: ModelsStore
    @EnvironmentObject private var training: TrainingCoordinator

    @State private var startError: String?
    @State private var confirmStop = false

    private var model: WorldModelRecord? {
        if let id = store.trainingModelID, let found = models.model(id: id) { return found }
        return models.models.first
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                picker
                if let model {
                    controls(model)
                    statusCard
                    metrics
                    SectionCard(title: "Loss history", subtitle: "Training loss per step, with held-out validation points marked", systemImage: "chart.xyaxis.line") {
                        LossChart(points: training.progress.history)
                    }
                    checkpointSection(model)
                    datasetSection(model)
                } else {
                    EmptyStateView(
                        systemImage: "chart.line.uptrend.xyaxis",
                        title: "Nothing to train yet",
                        message: "Create a world model in the World Models tab and give it recordings to learn from.",
                        actionTitle: "Go to World Models",
                        action: { store.selectTab(.models) })
                        .frame(height: 420)
                }
            }
            .padding(18)
        }
        .navigationTitle("Training")
        .confirmationDialog("Stop training?", isPresented: $confirmStop) {
            Button("Stop and Save Checkpoint", role: .destructive) {
                training.cancel()
            }
            Button("Keep Training", role: .cancel) {}
        } message: {
            Text("Training writes a resumable checkpoint before it stops, so nothing already learned is lost.")
        }
    }

    private var picker: some View {
        HStack(spacing: 12) {
            Text("Model")
                .foregroundStyle(.secondary)
            Picker("", selection: Binding(
                get: { store.trainingModelID ?? models.models.first?.id },
                set: { store.trainingModelID = $0 })) {
                ForEach(models.models) { model in
                    Text(model.name).tag(Optional(model.id))
                }
            }
            .labelsHidden()
            .frame(maxWidth: 320)
            .disabled(training.isRunning)

            if let model {
                Badge(text: model.configuration.resolution.rawValue, tint: .secondary)
                Badge(text: "\(model.recordingIDs.count) recordings", tint: .secondary)
                Badge(text: model.isTrained ? "trained before" : "untrained", tint: model.isTrained ? .green : .orange)
            }
            Spacer()
        }
    }

    private func controls(_ model: WorldModelRecord) -> some View {
        let sources = models.sources(for: model, library: library).filter { $0.manifest.isStructurallyValid }
        let resumable = CheckpointStore.latest(
            modelID: model.id,
            configurationFingerprint: model.configuration.architectureFingerprint,
            datasetFingerprint: DatasetBuilder.fingerprint(configuration: model.configuration, sources: sources),
            directories: store.directories) != nil
        let issues = model.configuration.validationIssues()
        let blocked = issues.first ?? (sources.isEmpty ? "This model has no usable recordings selected." : nil)

        return SectionCard(title: "Run control", subtitle: blocked, systemImage: "play.circle") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    Button {
                        startError = store.startTraining(modelID: model.id, resume: false)
                    } label: {
                        Label("Start Training", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(training.isRunning || blocked != nil)

                    Button {
                        startError = store.startTraining(modelID: model.id, resume: true)
                    } label: {
                        Label("Resume", systemImage: "arrow.clockwise")
                    }
                    .disabled(training.isRunning || blocked != nil || !resumable)
                    .help(resumable
                          ? "Continue from the newest checkpoint that matches this configuration and dataset"
                          : "No checkpoint matches this configuration and dataset yet")

                    Button {
                        training.pause()
                    } label: {
                        Label("Pause", systemImage: "pause.fill")
                    }
                    .disabled(!training.canPause)

                    Button(role: .destructive) {
                        confirmStop = true
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .disabled(!training.isRunning)

                    Spacer()

                    if let checkpoint = training.progress.checkpointID {
                        Text("Latest checkpoint \(checkpoint.uuidString.prefix(8))")
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
                if let startError {
                    Label(startError, systemImage: "exclamationmark.triangle.fill")
                        .font(.callout)
                        .foregroundStyle(.orange)
                }
                if let error = training.progress.error {
                    Label(error, systemImage: "xmark.octagon.fill")
                        .font(.callout)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var statusCard: some View {
        let progress = training.progress
        let fraction = progressFraction(progress)
        return SectionCard(title: "Status", subtitle: progress.message, systemImage: "waveform.path.ecg") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Badge(
                        text: progress.phase.rawValue,
                        tint: phaseTint(progress.phase),
                        systemImage: progress.phase.isBusy ? "bolt.fill" : "circle")
                    if progress.epochs > 0 {
                        Badge(text: "Epoch \(progress.epoch)/\(progress.epochs)", tint: .secondary)
                    }
                    if progress.stepsPerEpoch > 0 {
                        Badge(text: "Step \(progress.stepWithinEpoch)/\(progress.stepsPerEpoch)", tint: .secondary)
                    }
                    Spacer()
                    Text(Format.duration(progress.elapsedSeconds))
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                ProgressView(value: fraction)
                    .progressViewStyle(.linear)
            }
        }
    }

    private func progressFraction(_ progress: TrainingProgress) -> Double {
        guard progress.epochs > 0 else { return 0 }
        let withinEpoch = progress.stepsPerEpoch > 0
            ? Double(progress.stepWithinEpoch) / Double(progress.stepsPerEpoch)
            : 0
        return min(1, max(0, (Double(progress.epoch) + withinEpoch) / Double(progress.epochs)))
    }

    private func phaseTint(_ phase: TrainingPhase) -> Color {
        switch phase {
        case .failed: return .red
        case .cancelled, .paused: return .orange
        case .complete: return .green
        case .idle: return .secondary
        default: return .accentColor
        }
    }

    private var metrics: some View {
        let progress = training.progress
        let showAdvanced = store.settings.preferences.showAdvancedMetrics
        return SectionCard(title: "Live metrics", systemImage: "speedometer") {
            VStack(alignment: .leading, spacing: 10) {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 10)], spacing: 10) {
                    StatTile(
                        title: "Training loss",
                        value: progress.trainingLoss.map { Format.decimal(Double($0)) } ?? "—",
                        detail: progress.step > 0 ? "\(Format.integer(progress.step)) steps" : "not started",
                        tint: .accentColor)
                    StatTile(
                        title: "Validation loss",
                        value: progress.validationLoss.map { Format.decimal(Double($0)) } ?? "—",
                        detail: "held-out footage",
                        tint: .orange)
                    StatTile(
                        title: "Best validation",
                        value: progress.bestValidationLoss.map { Format.decimal(Double($0)) } ?? "—",
                        detail: "drives the best checkpoint",
                        tint: .green)
                    StatTile(
                        title: "Steps / second",
                        value: progress.stepsPerSecond > 0 ? Format.rate(progress.stepsPerSecond) : "—",
                        detail: "measured throughput",
                        tint: .purple)
                    StatTile(
                        title: "Elapsed",
                        value: Format.duration(progress.elapsedSeconds),
                        detail: "wall clock",
                        tint: .secondary)
                    StatTile(
                        title: "Epoch",
                        value: "\(progress.epoch)/\(max(1, progress.epochs))",
                        detail: progress.samplesPerEpoch > 0 ? "\(Format.integer(progress.samplesPerEpoch)) windows per epoch" : nil,
                        tint: .secondary)
                    StatTile(
                        title: "Dataset frames",
                        value: Format.integer(progress.datasetFrames),
                        detail: progress.validationFrames > 0 ? "\(Format.integer(progress.validationFrames)) held out" : "no held-out frames",
                        tint: .secondary)
                    if showAdvanced {
                        StatTile(
                            title: "Learning rate",
                            value: Format.decimal(Double(progress.learningRate), places: 5),
                            detail: "AdamW",
                            tint: .secondary)
                        StatTile(
                            title: "Gradient norm",
                            value: progress.gradientNorm.map { Format.decimal(Double($0), places: 3) } ?? "—",
                            detail: "before clipping",
                            tint: .secondary)
                        StatTile(
                            title: "MLX memory",
                            value: Format.bytes(Int64(progress.activeMemory)),
                            detail: "peak cache \(Format.bytes(Int64(progress.cacheMemory)))",
                            tint: .secondary)
                    }
                }
            }
        }
    }

    private func checkpointSection(_ model: WorldModelRecord) -> some View {
        let list = models.checkpoints(for: model.id)
        return SectionCard(
            title: "Saved checkpoints",
            subtitle: "Written every epoch, every 200 steps, and on pause or stop. The newest four plus the best are kept.",
            systemImage: "externaldrive"
        ) {
            if list.isEmpty {
                Text("None yet.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(list.prefix(8)) { checkpoint in
                        HStack(spacing: 8) {
                            Badge(text: checkpoint.kind.rawValue, tint: checkpoint.kind == .best ? .yellow : .secondary)
                            Text(checkpoint.label).font(.callout)
                            Text(checkpoint.createdAt.formatted(date: .omitted, time: .standard))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            if let validation = checkpoint.validationLoss {
                                Text("val \(Format.decimal(Double(validation)))")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            if model.trainedCheckpointID == checkpoint.id {
                                Badge(text: "Run", tint: .green)
                            }
                        }
                    }
                }
            }
        }
    }

    private func datasetSection(_ model: WorldModelRecord) -> some View {
        let sources = models.sources(for: model, library: library)
        let cacheBytes = models.cacheBytes(for: model.id)
        return SectionCard(title: "Prepared data", systemImage: "internaldrive") {
            VStack(alignment: .leading, spacing: 6) {
                KeyValueRow(key: "Recordings selected", value: "\(model.recordingIDs.count) (\(sources.count) resolvable)")
                KeyValueRow(key: "Cache on disk", value: Format.bytes(cacheBytes))
                KeyValueRow(key: "Resolution / FPS", value: "\(model.configuration.width) × \(model.configuration.height) at \(Format.rate(model.configuration.targetFPS)) FPS")
                KeyValueRow(key: "Sequence length", value: "\(model.configuration.sequenceLength) frames")
                KeyValueRow(key: "Configuration fingerprint", value: model.configuration.architectureFingerprint, monospaced: true)
                Text("The decoded frame and control cache is rebuilt automatically whenever the selection, resolution, FPS or architecture changes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
