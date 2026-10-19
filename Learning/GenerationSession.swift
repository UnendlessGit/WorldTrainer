import CoreGraphics
import Foundation
import MLX

/// Value snapshot published from the generation queue to the interface.
struct GenerationSnapshot: Sendable {
    enum Kind: Sendable {
        case ready
        case frame
        case finished
        case failed(String)
    }

    var kind: Kind
    var bytes: [UInt8] = []
    var width = 0
    var height = 0
    var frameIndex = 0
    var measuredFPS = 0.0
    var headroomFPS = 0.0
    var stepMilliseconds = 0.0
    var elapsedSeconds = 0.0
}

/// The generation loop. Confined to one serial queue: MLX arrays and the model
/// never leave it, and only value snapshots cross back to the main actor.
final class GenerationSession {
    private let queue = DispatchQueue(label: "com.worldtrainer.generation", qos: .userInitiated)
    private let lock = NSLock()
    private var runningFlag = false

    private let inputBox: ControlStateBox
    private let capabilities: ActionCapabilities
    private let region: CodableRect
    private let publish: @Sendable (GenerationSnapshot) -> Void

    private var engine: GenerationEngine?
    private var targetFPS: Double = 20
    private var started = Date()
    private var lastStatsPublish = Date.distantPast

    init(
        inputBox: ControlStateBox,
        capabilities: ActionCapabilities,
        region: CodableRect,
        publish: @escaping @Sendable (GenerationSnapshot) -> Void
    ) {
        self.inputBox = inputBox
        self.capabilities = capabilities
        self.region = region
        self.publish = publish
    }

    var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return runningFlag
    }

    private func setRunning(_ value: Bool) {
        lock.lock()
        runningFlag = value
        lock.unlock()
    }

    func setTargetFPS(_ value: Double) {
        lock.lock()
        targetFPS = max(1, min(120, value))
        lock.unlock()
    }

    /// Loads the checkpoint and warms the model up. Runs off the main actor so
    /// the interface stays responsive while weights are read.
    func prepare(
        checkpoint: TrainingCheckpoint,
        checkpointDirectory: URL,
        seed: GenerationSeed,
        memoryLimitGB: Double,
        targetFPS: Double
    ) {
        setRunning(true)
        setTargetFPS(targetFPS)
        queue.async { [self] in
            do {
                Memory.memoryLimit = Int(max(2, memoryLimitGB) * 1_073_741_824)
                let engine = try GenerationEngine(
                    checkpoint: checkpoint,
                    checkpointDirectory: checkpointDirectory,
                    seed: seed,
                    capabilities: capabilities)
                self.engine = engine
                started = Date()
                publish(GenerationSnapshot(
                    kind: .ready,
                    width: checkpoint.configuration.width,
                    height: checkpoint.configuration.height))
                loop()
            } catch {
                setRunning(false)
                publish(GenerationSnapshot(kind: .failed(error.localizedDescription)))
            }
        }
    }

    func stop() {
        setRunning(false)
    }

    private func loop() {
        guard let engine else { return }
        var frameIndex = 0
        var previousTick = CFAbsoluteTimeGetCurrent()
        var pacedFPS = 0.0
        var headroomFPS = 0.0
        while isRunning {
            let tickStart = CFAbsoluteTimeGetCurrent()
            let state = inputBox.consumeForTick()
            let action = ActionLayout.values(for: state, capabilities: capabilities, region: region)
            let bytes = engine.step(action: action)
            let compute = CFAbsoluteTimeGetCurrent() - tickStart
            let interval = tickStart - previousTick
            previousTick = tickStart
            frameIndex += 1
            if interval > 0, interval < 10 {
                pacedFPS = pacedFPS == 0 ? 1 / interval : pacedFPS * 0.8 + (1 / interval) * 0.2
            }
            if compute > 0 {
                headroomFPS = headroomFPS == 0 ? 1 / compute : headroomFPS * 0.8 + (1 / compute) * 0.2
            }
            let now = Date()
            let elapsed = now.timeIntervalSince(started)
            let statsDue = now.timeIntervalSince(lastStatsPublish) > 0.25
            if statsDue { lastStatsPublish = now }
            publish(GenerationSnapshot(
                kind: .frame,
                bytes: bytes,
                width: engine.configuration.width,
                height: engine.configuration.height,
                frameIndex: frameIndex,
                measuredFPS: statsDue ? pacedFPS : 0,
                headroomFPS: statsDue ? headroomFPS : 0,
                stepMilliseconds: statsDue ? engine.averageStepSeconds * 1000 : 0,
                elapsedSeconds: elapsed))
            lock.lock()
            let interval2 = 1 / max(1, targetFPS)
            lock.unlock()
            let remaining = interval2 - (CFAbsoluteTimeGetCurrent() - tickStart)
            if remaining > 0 {
                Thread.sleep(forTimeInterval: remaining)
            }
        }
        publish(GenerationSnapshot(kind: .finished, frameIndex: frameIndex))
    }
}
