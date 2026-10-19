import AVFoundation
import CoreGraphics
import CoreImage
import CoreVideo
import Foundation

struct VideoTrackInfo {
    var asset: AVURLAsset
    var track: AVAssetTrack
    var duration: Double
    var naturalSize: CGSize
    var preferredTransform: CGAffineTransform
    var nominalFrameRate: Float

    var displaySize: CGSize {
        let transformed = naturalSize.applying(preferredTransform)
        return CGSize(width: abs(transformed.width), height: abs(transformed.height))
    }
}

enum VideoFramesError: LocalizedError {
    case noVideoTrack
    case unreadable(String)
    case decodeFailed(String)
    case renderFailed
    case empty

    var errorDescription: String? {
        switch self {
        case .noVideoTrack: return "The package has no readable video track."
        case .unreadable(let detail): return "The video could not be read: \(detail)"
        case .decodeFailed(let detail): return "Video decoding failed: \(detail)"
        case .renderFailed: return "A video frame could not be converted for the model."
        case .empty: return "The video contains no decodable frames."
        }
    }
}

/// AVFoundation based frame extraction for recording packages.
///
/// Frames are decoded sequentially, resized to the model resolution with
/// aspect-fit letterboxing, and delivered as tightly packed RGB `UInt8` bytes.
/// The stream is deliberately causal: a decoded frame remains current until the
/// next frame's presentation time, so static recordings still produce one
/// sample per configured interval.
enum VideoFrames {
    static func loadTrack(url: URL) async throws -> VideoTrackInfo {
        let asset = AVURLAsset(url: url)
        do {
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let track = tracks.first else { throw VideoFramesError.noVideoTrack }
            let duration = try await asset.load(.duration)
            let naturalSize = try await track.load(.naturalSize)
            let transform = try await track.load(.preferredTransform)
            let fps = try await track.load(.nominalFrameRate)
            return VideoTrackInfo(
                asset: asset, track: track,
                duration: max(0, CMTimeGetSeconds(duration)),
                naturalSize: naturalSize,
                preferredTransform: transform,
                nominalFrameRate: fps)
        } catch let error as VideoFramesError {
            throw error
        } catch {
            throw VideoFramesError.unreadable(error.localizedDescription)
        }
    }

    /// Extract one preview image. `appliesPreferredTrackTransform` keeps the
    /// generated image upright.
    static func image(url: URL, at seconds: Double, maximumSize: CGSize? = nil) async throws -> CGImage {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = CMTime(seconds: 1, preferredTimescale: 600)
        if let maximumSize { generator.maximumSize = maximumSize }
        do {
            let result = try await generator.image(at: CMTime(seconds: max(0, seconds), preferredTimescale: 600))
            return result.image
        } catch {
            throw VideoFramesError.decodeFailed(error.localizedDescription)
        }
    }

    /// Decode `start..<end` at a fixed rate and stream RGB frames to `onFrame`.
    /// Returns the number of emitted frames.
    @discardableResult
    static func streamRGB(
        info: VideoTrackInfo,
        start: Double,
        end: Double,
        fps: Double,
        width: Int,
        height: Int,
        onFrame: (Int, Data) throws -> Void,
        onProgress: ((Double) -> Void)? = nil
    ) throws -> Int {
        guard end > start, fps > 0, width > 0, height > 0 else { throw VideoFramesError.empty }
        let interval = 1 / fps
        let span = end - start
        let sampleCount = max(1, Int(floor(span * fps + 1e-6)) + 1)

        let reader = try AVAssetReader(asset: info.asset)
        reader.timeRange = CMTimeRange(
            start: CMTime(seconds: start, preferredTimescale: 600),
            duration: CMTime(seconds: span, preferredTimescale: 600))
        let output = AVAssetReaderTrackOutput(
            track: info.track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw VideoFramesError.decodeFailed("The video track cannot be decoded.") }
        reader.add(output)
        guard reader.startReading() else {
            throw VideoFramesError.decodeFailed(reader.error?.localizedDescription ?? "The reader did not start.")
        }

        let context = CIContext(options: [.cacheIntermediates: false])
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let destinationBounds = CGRect(x: 0, y: 0, width: width, height: height)
        var pool: CVPixelBufferPool?
        let poolAttributes: [String: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: 3,
        ]
        let bufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary,
        ]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttributes as CFDictionary, bufferAttributes as CFDictionary, &pool)
        guard let pool else { throw VideoFramesError.renderFailed }

        let orientation = exifOrientation(for: info.preferredTransform)
        var emitted = 0
        var nextSampleTime = start
        var currentBuffer: CVPixelBuffer?
        var firstPTS: CMTime?
        var progressTick = 0

        func emit(_ buffer: CVPixelBuffer) throws {
            var destination: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &destination) == kCVReturnSuccess,
                  let destination else {
                throw VideoFramesError.renderFailed
            }
            try render(buffer, orientation: orientation, into: destination, bounds: destinationBounds,
                       context: context, colorSpace: colorSpace)
            let data = try rgbData(from: destination)
            try onFrame(emitted, data)
            emitted += 1
            progressTick += 1
            if progressTick % 24 == 0, let onProgress {
                onProgress(min(1, Double(emitted) / Double(sampleCount)))
            }
        }

        while let sample = output.copyNextSampleBuffer(), let buffer = sample.imageBuffer {
            let pts = sample.presentationTimeStamp
            if firstPTS == nil { firstPTS = pts }
            guard let firstPTS else { continue }
            let t = start + max(0, CMTimeGetSeconds(pts - firstPTS))
            if currentBuffer == nil {
                currentBuffer = buffer
                while nextSampleTime <= t + 1e-9 && emitted < sampleCount {
                    try emit(buffer)
                    nextSampleTime += interval
                }
                continue
            }
            while nextSampleTime < t - 1e-9 && emitted < sampleCount {
                try emit(currentBuffer!)
                nextSampleTime += interval
            }
            currentBuffer = buffer
            while nextSampleTime <= t + 1e-9 && emitted < sampleCount {
                try emit(buffer)
                nextSampleTime += interval
            }
        }

        if reader.status == .failed {
            throw VideoFramesError.decodeFailed(reader.error?.localizedDescription ?? "Unknown decode failure.")
        }
        if let currentBuffer {
            while emitted < sampleCount {
                try emit(currentBuffer)
                nextSampleTime += interval
            }
        }
        guard emitted > 0 else { throw VideoFramesError.empty }
        onProgress?(1)
        return emitted
    }

    private static func render(
        _ source: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        into destination: CVPixelBuffer,
        bounds: CGRect,
        context: CIContext,
        colorSpace: CGColorSpace
    ) throws {
        let image = CIImage(cvPixelBuffer: source).oriented(orientation)
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else { throw VideoFramesError.renderFailed }
        let scale = min(bounds.width / extent.width, bounds.height / extent.height)
        let scaledWidth = extent.width * scale
        let scaledHeight = extent.height * scale
        let transform = CGAffineTransform(translationX: (bounds.width - scaledWidth) / 2,
                                          y: (bounds.height - scaledHeight) / 2)
            .scaledBy(x: scale, y: scale)
        let positioned = image.transformed(by: transform)
        let background = CIImage(color: .black).cropped(to: bounds)
        let composited = positioned.composited(over: background)
        context.render(composited, to: destination, bounds: bounds, colorSpace: colorSpace)
    }

    private static func rgbData(from buffer: CVPixelBuffer) throws -> Data {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw VideoFramesError.renderFailed }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        var data = Data(count: width * height * 3)
        data.withUnsafeMutableBytes { raw in
            guard let destination = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for y in 0..<height {
                let source = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
                let row = destination.advanced(by: y * width * 3)
                for x in 0..<width {
                    let pixel = source.advanced(by: x * 4)
                    let target = row.advanced(by: x * 3)
                    target[0] = pixel[2]
                    target[1] = pixel[1]
                    target[2] = pixel[0]
                }
            }
        }
        return data
    }

    static func exifOrientation(for transform: CGAffineTransform) -> CGImagePropertyOrientation {
        let a = transform.a, b = transform.b, c = transform.c, d = transform.d
        if a > 0.99 && b > -0.01 && b < 0.01 && c > -0.01 && c < 0.01 && d > 0.99 { return .up }
        if a > -0.01 && a < 0.01 && b > 0.99 && c < -0.99 && d > -0.01 && d < 0.01 { return .right }
        if a < -0.99 && b > -0.01 && b < 0.01 && c > -0.01 && c < 0.01 && d < -0.99 { return .down }
        if a > -0.01 && a < 0.01 && b < -0.99 && c > 0.99 && d > -0.01 && d < 0.01 { return .left }
        return .up
    }
}
