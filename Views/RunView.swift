import SwiftUI

struct RunView: View {
    @EnvironmentObject private var store: AppStore
    @EnvironmentObject private var library: LibraryStore
    @EnvironmentObject private var models: ModelsStore
    @EnvironmentObject private var run: GenerationCoordinator
    @EnvironmentObject private var settings: SettingsStore

    @State private var starting = false

    private var trainedModels: [WorldModelRecord] { models.trainedModels }

    private var selectedModel: WorldModelRecord? {
        if let id = store.runModelID, let model = models.model(id: id), model.isTrained { return model }
        return trainedModels.first
    }

    var body: some View {
        HSplitView {
            canvas
                .frame(minWidth: 520)
            controls
                .frame(minWidth: 320, idealWidth: 360, maxWidth: 460)
        }
        .navigationTitle("Run")
        .onAppear {
            if store.runModelID == nil { store.runModelID = trainedModels.first?.id }
        }
        .onDisappear {
            run.stop()
            run.capture.releaseAll()
        }
    }

    // MARK: - Canvas

    private var canvas: some View {
        VStack(spacing: 0) {
            ZStack {
                Color.black
                if let frame = run.frame {
                    Image(decorative: frame, scale: 1)
                        .resizable()
                        .interpolation(.none)
                        .aspectRatio(contentMode: .fit)
                } else {
                    placeholder
                }
                InputCaptureLayer(capture: run.capture)
                hud
                if run.phase == .running, !run.capture.isFocused {
                    VStack {
                        Spacer()
                        Text("Click the view to capture input · Command-. stops generation")
                            .font(.caption)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(.black.opacity(0.6), in: Capsule())
                            .foregroundStyle(.white)
                            .padding(.bottom, 14)
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.separator))
            .padding(14)

            statusStrip
        }
        .background(.background.secondary)
    }

    private var placeholder: some View {
        VStack(spacing: 8) {
            Image(systemName: "play.rectangle.on.rectangle")
                .font(.system(size: 30))
                .foregroundStyle(.tertiary)
            Text(run.phase == .preparing ? "Loading checkpoint and warming up memory…" : "No generation running")
                .font(.callout)
                .foregroundStyle(.secondary)
            if run.phase != .preparing {
                Text("Every frame shown here is produced by the selected model.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var hud: some View {
        VStack {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(run.phase == .running ? Color.green : Color.orange)
                            .frame(width: 7, height: 7)
                        Text(run.phase.label)
                            .font(.caption.weight(.semibold))
                    }
                    if run.phase == .running {
                        Text("\(Format.rate(run.measuredFPS)) FPS displayed · \(Format.rate(run.headroomFPS)) FPS max")
                            .font(.caption2.monospacedDigit())
                        Text("frame \(Format.integer(run.frameIndex)) · \(Format.duration(run.elapsedSeconds))")
                            .font(.caption2.monospacedDigit())
                    }
                }
                .padding(8)
                .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
                .foregroundStyle(.white)

                Spacer()

                if !run.detail.resolution.isEmpty {
                    VStack(alignment: .trailing, spacing: 3) {
                        Text(run.detail.resolution)
                            .font(.caption2.monospacedDigit())
                        Text("\(Format.decimal(run.stepMilliseconds, places: 1)) ms / frame")
                            .font(.caption2.monospacedDigit())
                        Text(run.detail.modelName)
                            .font(.caption2)
                            .lineLimit(1)
                    }
                    .padding(8)
                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
                    .foregroundStyle(.white)
                }
            }
            Spacer()
        }
        .padding(12)
        .allowsHitTesting(false)
    }

    private var statusStrip: some View {
        HStack(spacing: 16) {
            Label("\(Format.integer(run.frameIndex)) frames generated", systemImage: "film.stack")
            Label("\(Format.rate(run.measuredFPS)) FPS shown", systemImage: "speedometer")
            Label("\(Format.rate(run.headroomFPS)) FPS max", systemImage: "bolt")
            Label("\(Format.decimal(run.stepMilliseconds, places: 1)) ms/step", systemImage: "timer")
            Label(Format.duration(run.elapsedSeconds), systemImage: "clock")
            Spacer()
            if !run.detail.seedRecording.isEmpty {
                Text("Memory warmed from \(run.detail.seedRecording) · \(run.detail.warmupFrames) frames")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .padding(.bottom, 12)
    }

    // MARK: - Controls

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if trainedModels.isEmpty {
                    EmptyStateView(
                        systemImage: "brain",
                        title: "No trained model",
                        message: "Train a world model first. Run generates frames from saved weights, it never replays recorded footage.",
                        actionTitle: "Go to Training",
                        action: { store.selectTab(.training) })
                } else {
                    sessionCard
                    performanceCard
                    inputCard
                    disclosureCard
                }
            }
            .padding(14)
        }
    }

    private var sessionCard: some View {
        SectionCard(title: "Session", systemImage: "play.circle") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Model", selection: Binding(
                    get: { store.runModelID ?? trainedModels.first?.id },
                    set: { store.runModelID = $0 })) {
                    ForEach(trainedModels) { model in
                        Text(model.name).tag(Optional(model.id))
                    }
                }
                .disabled(run.isActive)

                if let model = selectedModel {
                    KeyValueRow(key: "Checkpoint", value: model.checkpointSummary)
                    if let trainedAt = model.trainedAt {
                        KeyValueRow(key: "Trained", value: trainedAt.formatted(date: .abbreviated, time: .shortened))
                    }
                    KeyValueRow(key: "Resolution", value: "\(model.configuration.width) × \(model.configuration.height)")
                    KeyValueRow(key: "Model FPS", value: Format.rate(model.configuration.targetFPS))
                    KeyValueRow(key: "Parameters", value: Format.integer(model.configuration.estimatedParameterCount()))
                }

                HStack(spacing: 8) {
                    Button {
                        guard let model = selectedModel else { return }
                        starting = true
                        Task {
                            await store.startRun(modelID: model.id)
                            starting = false
                        }
                    } label: {
                        Label(run.isActive ? "Restart" : "Start Generation", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(starting || selectedModel == nil)

                    Button(role: .destructive) {
                        run.stop()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .disabled(!run.isActive)
                }

                if starting || run.phase == .preparing {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Reading the checkpoint and warming up temporal memory…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                if let error = run.error ?? store.runError {
                    Label(error, systemImage: "xmark.octagon.fill")
                        .font(.callout)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var performanceCard: some View {
        SectionCard(
            title: "Performance",
            subtitle: "Generation runs as fast as the model allows and is paced down to the target rate.",
            systemImage: "speedometer"
        ) {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Target rate")
                    Spacer()
                    Text("\(Format.rate(run.targetFPS)) FPS")
                        .font(.callout.monospacedDigit())
                }
                Slider(value: Binding(
                    get: { run.targetFPS },
                    set: { run.targetFPS = $0 }),
                       in: 5...120, step: 1)
                Text("The model keeps up as fast as it can; the target rate only paces the output. Watch the measured rate above to see how much headroom is left.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Warm-up frames")
                    Spacer()
                    Stepper("\(settings.preferences.runWarmupFrames)", value: Binding(
                        get: { settings.preferences.runWarmupFrames },
                        set: { settings.preferences.runWarmupFrames = $0 }),
                            in: 4...120, step: 4)
                        .labelsHidden()
                        .frame(width: 110)
                }

                Divider()

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 8)], spacing: 8) {
                    StatTile(
                        title: "Shown",
                        value: run.measuredFPS > 0 ? Format.rate(run.measuredFPS) : "—",
                        detail: "paced to target",
                        tint: run.measuredFPS >= 15 ? .green : .orange)
                    StatTile(
                        title: "Max rate",
                        value: run.headroomFPS > 0 ? Format.rate(run.headroomFPS) : "—",
                        detail: "model headroom",
                        tint: .purple)
                    StatTile(
                        title: "Per frame",
                        value: run.stepMilliseconds > 0 ? "\(Format.decimal(run.stepMilliseconds, places: 1)) ms" : "—",
                        detail: "model forward pass",
                        tint: .secondary)
                    StatTile(
                        title: "Frames",
                        value: Format.integer(run.frameIndex),
                        detail: "no duration limit",
                        tint: .secondary)
                }
                Text("Warm-up frames come from the seed recording and only fill the model's memory. Everything after the last warm-up frame is generated.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var inputCard: some View {
        let capabilities = selectedModel.map { models.capabilities(for: $0, library: library) } ?? ActionCapabilities()
        let state = run.capture.state
        return SectionCard(title: "Input", subtitle: run.capture.message, systemImage: "keyboard") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Badge(
                        text: run.capture.isFocused ? "Capturing" : "Not capturing",
                        tint: run.capture.isFocused ? .green : .secondary,
                        systemImage: run.capture.isFocused ? "dot.radiowaves.left.and.right" : "pause.circle")
                    Text("\(Format.integer(run.capture.eventCount)) events")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Release All") { run.capture.releaseAll() }
                        .controlSize(.small)
                }

                Divider()
                Text("Enabled by this model's recordings")
                    .font(.subheadline.weight(.semibold))
                CapabilitySummary(
                    capabilities: capabilities,
                    emptyMessage: "This model's recordings contain no input events, so Run only generates video.")

                if !capabilities.isEmpty {
                    Divider()
                    Text("Live")
                        .font(.subheadline.weight(.semibold))
                    VStack(alignment: .leading, spacing: 4) {
                        KeyValueRow(key: "Held keys", value: state.heldKeys.isEmpty
                                    ? "none"
                                    : VirtualKeyNames.sortedNames(state.heldKeys).joined(separator: " + "))
                        KeyValueRow(key: "Mouse buttons", value: state.heldButtons.isEmpty
                                    ? "none"
                                    : state.heldButtons.sorted().map(MouseButtonNames.name(for:)).joined(separator: ", "))
                        KeyValueRow(key: "Pointer", value: state.pointerValid
                                    ? String(format: "%.2f, %.2f", state.pointerX, state.pointerY)
                                    : "not moved yet")
                        KeyValueRow(key: "Relative mouse", value: String(format: "%.1f, %.1f", state.mouseDeltaX, state.mouseDeltaY))
                        KeyValueRow(key: "Scroll", value: String(format: "%.1f, %.1f", state.scrollX, state.scrollY))
                        KeyValueRow(key: "Modifiers", value: modifierText(state.modifiers))
                    }
                }
            }
        }
    }

    private func modifierText(_ flags: UInt64) -> String {
        var parts: [String] = []
        if flags & 0x0002_0000 != 0 { parts.append("Shift") }
        if flags & 0x0004_0000 != 0 { parts.append("Control") }
        if flags & 0x0008_0000 != 0 { parts.append("Option") }
        if flags & 0x0010_0000 != 0 { parts.append("Command") }
        return parts.isEmpty ? "none" : parts.joined(separator: " + ")
    }

    private var disclosureCard: some View {
        SectionCard(title: "What Run does", systemImage: "info.circle") {
            VStack(alignment: .leading, spacing: 6) {
                Text("Frames are generated autoregressively from the model's own previous output. Recurrent memory persists for the whole session, so the world does not reset on a timer.")
                Text("Input is limited to the channels the selected recordings actually demonstrated. Keys and buttons outside that set are ignored rather than guessed.")
                Text("Quality is bounded by the training data: the model learns the games it saw, and short training runs produce blurrier, less coherent frames. Raise the resolution, channel counts and epochs to trade speed for fidelity.")
                Text("Command shortcuts still reach the app, so Command-Q quits and Command-. stops generation. Everything else is sent to the model while the view has focus, and released the moment it loses focus.")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }
}
