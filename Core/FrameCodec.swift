import CoreGraphics
import Foundation

/// Conversions between model tensors and displayable images.
///
/// The Run tab letterboxes with the same rule the dataset builder uses, so a
/// seed frame is preprocessed exactly like the frames the model was trained on.
enum FrameCodec {
    static let colorSpace = CGColorSpaceCreateDeviceRGB()

    /// Aspect-fit `image` into a `width`×`height` RGB buffer over black.
    static func rgbBytes(from image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        guard width > 0, height > 0 else { throw WorldTrainerError.generation("Invalid frame size.") }
        var bytes = [UInt8](repeating: 0, count: width * height * 3)
        let rendered: Bool = bytes.withUnsafeMutableBytes { raw in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 3,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.interpolationQuality = .high
            let scale = min(CGFloat(width) / CGFloat(max(1, image.width)),
                            CGFloat(height) / CGFloat(max(1, image.height)))
            let drawWidth = CGFloat(image.width) * scale
            let drawHeight = CGFloat(image.height) * scale
            context.draw(image, in: CGRect(
                x: (CGFloat(width) - drawWidth) / 2,
                y: (CGFloat(height) - drawHeight) / 2,
                width: drawWidth,
                height: drawHeight))
            return true
        }
        guard rendered else {
            throw WorldTrainerError.generation("A frame could not be prepared for the model.")
        }
        return bytes
    }

    /// Nearest-neighbour upscale of a model frame so the generated image stays
    /// crisp instead of blurry when shown larger than the model resolution.
    static func image(fromRGB bytes: [UInt8], width: Int, height: Int) -> CGImage? {
        guard bytes.count >= width * height * 3 else { return nil }
        guard let data = CFDataCreate(kCFAllocatorDefault, bytes, width * height * 3),
              let provider = CGDataProvider(data: data) else { return nil }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 24,
            bytesPerRow: width * 3,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent)
    }

    static func image(fromRGB data: Data, width: Int, height: Int) -> CGImage? {
        image(fromRGB: [UInt8](data), width: width, height: height)
    }
}
