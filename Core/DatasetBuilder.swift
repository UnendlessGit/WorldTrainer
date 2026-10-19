import AVFoundation
import CryptoKit
import Foundation

/// Bounded sequential writer used for multi-gigabyte dataset caches.
final class BufferedFileWriter {
    private let handle: FileHandle
    private var buffer = Data()
    private let threshold: Int
    private(set) var count = 0

    init(url: URL, threshold: Int = 4 * 1024 * 1024) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 0)
        self.threshold = threshold
        buffer.reserveCapacity(threshold)
    }

    func append(_ data: Data) throws {
        buffer.append(data)
        count += 1
        if buffer.count >= threshold { try flush() }
    }

    func append(_ values: [Float]) throws {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        try append(data)
    }

    func flush() throws {
        guard !buffer.isEmpty else { return }
        try handle.write(contentsOf: buffer)
        buffer.removeAll(keepingCapacity: true)
    }

    func finish() throws {
        try flush()
        try handle.close()
    }
}

struct DatasetRecordingSource: Sendable {
    var recordingID: UUID
    var name: String
    var directory: URL
    var manifest: RecordingManifest
    var videoDuration: Double?
}

enum DatasetBuilder {
    static func fingerprint(configuration: WorldModelConfiguration, sources: [DatasetRecordingSource]) -> String {
        struct Identity: Encodable {
            var resolution: String
            var targetFPS: Double
            var architecture: String
            var recordings: [RecordingIdentity]
        }
        struct RecordingIdentity: Encodable {
            var id: String
            var hostStart: UInt64
            var duration: Double
            var trimStart: Double
            var trimEnd: Double
            var eventCount: Int
        }
        let identity = Identity(
            resolution: configuration.resolution.rawValue,
            targetFPS: configuration.targetFPS,
            architecture: configuration.architectureFingerprint,
            recordings: sources.map {
                RecordingIdentity(
                    id: $0.recordingID.uuidString,
                    hostStart: $0.manifest.hostStartNanos,
                    duration: $0.manifest.duration,
                    trimStart: $0.manifest.effectiveTrimStart,
                    trimEnd: $0.manifest.effectiveTrimEnd,
                    eventCount: $0.manifest.eventCount)
            }.sorted { $0.id < $1.id })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = (try? encoder.encode(identity)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    @discardableResult
    static func build(
        sources: [DatasetRecordingSource],
        configuration: WorldModelConfiguration,
        fingerprint: String,
        directory: URL,
        progress: @escaping (Double, String) -> Void,
        checkCancellation: () throws -> Void
    ) throws -> DatasetCacheManifest {
        guard !sources.isEmpty else { throw WorldTrainerError.invalidData("Select at least one usable recording before training.") }
        let manager = FileManager.default
        try? manager.removeItem(at: directory)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            let manifest = try buildInternal(
                sources: sources, configuration: configuration,
                fingerprint: fingerprint, directory: directory,
                progress: progress, checkCancellation: checkCancellation)
            return manifest
        } catch {
            try? manager.removeItem(at: directory)
            throw error
        }
    }

    private static func buildInternal(
        sources: [DatasetRecordingSource],
        configuration: WorldModelConfiguration,
        fingerprint: String,
        directory: URL,
        progress: @escaping (Double, String) -> Void,
        checkCancellation: () throws -> Void
    ) throws -> DatasetCacheManifest {
        let width = configuration.width
        let height = configuration.height
        let fps = configuration.targetFPS
        let bytesPerFrame = width * height * 3
        let bytesPerAction = ActionLayout.count * MemoryLayout<Float>.size
        let frames = try BufferedFileWriter(url: directory.appendingPathComponent("frames.bin"))
        let actions = try BufferedFileWriter(url: directory.appendingPathComponent("actions.bin"))

        var entries: [DatasetCacheManifest.RecordingEntry] = []
        var validationAssignments = validationMap(sources: sources, configuration: configuration)

        var totalDuration = 0.0
        for source in sources {
            let start = source.manifest.effectiveTrimStart
            let end = source.manifest.effectiveTrimEnd
            totalDuration += max(0, end - start)
        }
        var completedDuration = 0.0

        for source in sources {
            try checkCancellation()
            let start = source.manifest.effectiveTrimStart
            let end = max(start, source.manifest.effectiveTrimEnd)
            guard end > start else {
                validationAssignments[source.recordingID] = false
                continue
            }
            let expectedFrames = max(1, Int(floor((end - start) * fps + 1e-6)) + 1)
            let frameOffset = frames.count
            let actionOffset = actions.count

            let eventURL = source.directory.appendingPathComponent(source.manifest.eventFile)
            let cursor = try EventFile.EventCursor(url: eventURL)
            var cursorState = cursor
            var accumulator = ActionAccumulator(
                region: source.manifest.globalRect ?? .zero,
                initialPointer: nil)
            let startNanos = try hostNanos(source.manifest, seconds: start)
            while let event = cursorState.peek(), event.timestampNanos <= startNanos {
                accumulator.consume(event)
                _ = cursorState.next()
            }
            accumulator.endTick()

            progress(completedDuration / max(0.001, totalDuration),
                     "Decoding \(source.name) at \(Int(fps)) FPS…")

            let videoURL = source.directory.appendingPathComponent(source.manifest.videoFile)
            let info = try loadTrackSynchronously(url: videoURL)
            var emitted = 0
            try VideoFrames.streamRGB(
                info: info, start: start, end: end, fps: fps, width: width, height: height,
                onFrame: { index, data in
                    try checkCancellation()
                    try frames.append(data)
                    let frameEnd = min(end, start + Double(index + 1) / fps)
                    let endNanos = try hostNanos(source.manifest, seconds: frameEnd)
                    while let event = cursorState.peek(), event.timestampNanos < endNanos {
                        accumulator.consume(event)
                        _ = cursorState.next()
                    }
                    try actions.append(accumulator.actionValues())
                    accumulator.endTick()
                    emitted += 1
                },
                onProgress: { fraction in
                    let local = (end - start) * fraction
                    progress(min(0.999, (completedDuration + local) / max(0.001, totalDuration)),
                             "Decoding \(source.name) at \(Int(fps)) FPS…")
                })

            let actualFrames = emitted
            guard actualFrames > 1 else { continue }
            var trainStart = 0
            var trainCount = actualFrames
            var validationStart = actualFrames
            var validationCount = 0
            if sources.count == 1, configuration.validationFraction > 0, actualFrames >= configuration.sequenceLength + 8 {
                let validation = max(configuration.sequenceLength + 2,
                                     Int((Double(actualFrames) * configuration.validationFraction).rounded()))
                let train = actualFrames - validation
                if train >= configuration.sequenceLength + 2 {
                    trainStart = 0
                    trainCount = train
                    validationStart = train
                    validationCount = validation
                }
            } else if validationAssignments[source.recordingID] == true {
                trainStart = 0
                trainCount = 0
                validationStart = 0
                validationCount = actualFrames
            }
            entries.append(DatasetCacheManifest.RecordingEntry(
                recordingID: source.recordingID,
                name: source.name,
                frameOffset: frameOffset,
                frameCount: actualFrames,
                actionOffset: actionOffset,
                actionCount: actualFrames,
                trainFrameStart: frameOffset + trainStart,
                trainFrameCount: trainCount,
                validationFrameStart: frameOffset + validationStart,
                validationFrameCount: validationCount))
            completedDuration += max(0, end - start)
            progress(min(0.999, completedDuration / max(0.001, totalDuration)),
                     "Decoded \(source.name) · \(actualFrames) frames")
        }

        try frames.finish()
        try actions.finish()

        let manifest = DatasetCacheManifest(
            fingerprint: fingerprint,
            createdAt: Date(),
            configuration: configuration,
            recordingIDs: sources.map(\.recordingID),
            recordings: entries,
            totalFrames: frames.count,
            totalActions: actions.count,
            bytesPerFrame: bytesPerFrame,
            bytesPerAction: bytesPerAction)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("dataset.json"), options: .atomic)

        let trainFrames = manifest.trainingFrameCount
        guard trainFrames >= configuration.sequenceLength + 2 else {
            throw WorldTrainerError.invalidData(
                "The selected recordings produced only \(trainFrames) training frames. At least \(configuration.sequenceLength + 2) are needed. Select longer recordings or lower the sequence length.")
        }
        progress(1, "Dataset ready · \(trainFrames) training frames")
        return manifest
    }

    private static func validationMap(
        sources: [DatasetRecordingSource],
        configuration: WorldModelConfiguration
    ) -> [UUID: Bool] {
        var result: [UUID: Bool] = [:]
        for source in sources { result[source.recordingID] = false }
        guard configuration.validationFraction > 0, sources.count >= 3 else { return result }
        let threshold = UInt64(configuration.validationFraction * 1000)
        for source in sources {
            let digest = SHA256.hash(data: Data(source.recordingID.uuidString.utf8))
            let value = digest.prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
            result[source.recordingID] = value % 1000 < threshold
        }
        if !result.values.contains(true), let last = sources.last {
            result[last.recordingID] = true
        }
        return result
    }

    private static func hostNanos(_ manifest: RecordingManifest, seconds: Double) throws -> UInt64 {
        let nanos = seconds * 1_000_000_000
        guard seconds.isFinite, seconds >= 0, nanos < Double(UInt64.max - manifest.hostStartNanos) else {
            throw WorldTrainerError.invalidData("\(manifest.name) contains a timestamp outside the supported range.")
        }
        return manifest.hostStartNanos + UInt64(nanos)
    }

    /// Synchronous track lookup for the serial dataset worker. The modern
    /// async AVFoundation loaders are used by the inspector, where an async
    /// context is available.
    private static func loadTrackSynchronously(url: URL) throws -> VideoTrackInfo {
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw VideoFramesError.noVideoTrack
        }
        return VideoTrackInfo(
            asset: asset,
            track: track,
            duration: max(0, CMTimeGetSeconds(asset.duration)),
            naturalSize: track.naturalSize,
            preferredTransform: track.preferredTransform,
            nominalFrameRate: track.nominalFrameRate)
    }
}
