import AppKit
import CoreGraphics
import SwiftUI

// MARK: - Formatting

enum Format {
    static func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
    }

    static func duration(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds > 0 else { return "0:00" }
        let total = Int(seconds.rounded())
        let minutes = total / 60
        let remainder = total % 60
        if minutes >= 60 {
            return String(format: "%d:%02d:%02d", minutes / 60, minutes % 60, remainder)
        }
        return String(format: "%d:%02d", minutes, remainder)
    }

    static func decimal(_ value: Double, places: Int = 4) -> String {
        value.formatted(.number.precision(.fractionLength(places)))
    }

    static func integer(_ value: Int) -> String {
        value.formatted(.number.grouping(.automatic))
    }

    static func rate(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(1)))
    }

    static func percent(_ value: Double) -> String {
        (value * 100).formatted(.number.precision(.fractionLength(0))) + "%"
    }
}

// MARK: - Small building blocks

struct SectionCard<Content: View>: View {
    let title: String
    var subtitle: String?
    var systemImage: String?
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .foregroundStyle(.secondary)
                }
                Text(title)
                    .font(.headline)
                Spacer()
            }
            if let subtitle {
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }
}

struct StatTile: View {
    let title: String
    let value: String
    var detail: String?
    var tint: Color = .accentColor
    var systemImage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.caption2)
                }
                Text(title.uppercased())
                    .font(.caption2.weight(.semibold))
                    .tracking(0.4)
            }
            .foregroundStyle(tint)
            Text(value)
                .font(.title3.monospacedDigit().weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            if let detail {
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(10)
        .frame(minWidth: 118, maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
    }
}

struct Badge: View {
    let text: String
    var tint: Color = .accentColor
    var systemImage: String?

    var body: some View {
        HStack(spacing: 4) {
            if let systemImage {
                Image(systemName: systemImage).font(.caption2)
            }
            Text(text).font(.caption2.weight(.medium))
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(tint.opacity(0.16), in: Capsule())
        .foregroundStyle(tint)
    }
}

struct KeyValueRow: View {
    let key: String
    let value: String
    var monospaced = false

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(key)
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            Text(value)
                .font(monospaced ? .callout.monospacedDigit() : .callout)
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
        .font(.callout)
    }
}

struct EmptyStateView: View {
    let systemImage: String
    let title: String
    let message: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 34))
                .foregroundStyle(.tertiary)
            Text(title)
                .font(.title3.weight(.semibold))
            Text(message)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }
}

/// Line chart for loss history. Drawn directly so validation points can be
/// marked without pulling in a charting dependency.
struct LossChart: View {
    let points: [LossPoint]
    var height: CGFloat = 180

    private var trainingValues: [Float] { points.map(\.training) }
    private var validationValues: [(Int, Float)] {
        points.enumerated().compactMap { index, point in
            point.validation.map { (index, $0) }
        }
    }

    var body: some View {
        Canvas { context, size in
            guard points.count > 1 else { return }
            let allValues = trainingValues + validationValues.map(\.1)
            guard let minimum = allValues.min(), let maximum = allValues.max(), maximum > minimum else { return }
            let inset: CGFloat = 8
            let width = size.width - inset * 2
            let chartHeight = size.height - inset * 2

            func point(_ index: Int, _ value: Float) -> CGPoint {
                let x = inset + width * CGFloat(index) / CGFloat(max(1, points.count - 1))
                let normalized = CGFloat((value - minimum) / (maximum - minimum))
                return CGPoint(x: x, y: inset + chartHeight * (1 - normalized))
            }

            var gridPath = Path()
            for step in 0...4 {
                let y = inset + chartHeight * CGFloat(step) / 4
                gridPath.move(to: CGPoint(x: inset, y: y))
                gridPath.addLine(to: CGPoint(x: inset + width, y: y))
            }
            context.stroke(gridPath, with: .color(.secondary.opacity(0.15)), lineWidth: 1)

            var path = Path()
            for (index, value) in trainingValues.enumerated() {
                let position = point(index, value)
                if index == 0 { path.move(to: position) } else { path.addLine(to: position) }
            }
            context.stroke(path, with: .color(.accentColor), style: StrokeStyle(lineWidth: 1.6, lineJoin: .round))

            for (index, value) in validationValues {
                let position = point(index, value)
                context.fill(
                    Path(ellipseIn: CGRect(x: position.x - 3, y: position.y - 3, width: 6, height: 6)),
                    with: .color(.orange))
            }

            let label = "\(Format.decimal(Double(minimum), places: 4)) – \(Format.decimal(Double(maximum), places: 4))"
            context.draw(
                Text(label).font(.caption2).foregroundStyle(.secondary),
                at: CGPoint(x: size.width - 8, y: 10), anchor: .trailing)
        }
        .frame(height: height)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .overlay(alignment: .center) {
            if points.count < 2 {
                Text("Loss history appears once training steps complete.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

/// Reports the window hosting a view so Run can drive first-responder input.
struct WindowAccessor: NSViewRepresentable {
    let onWindow: (NSWindow?) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { onWindow(view.window) }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async { onWindow(nsView.window) }
    }
}

struct ThumbnailView: View {
    let url: URL?
    var size: CGFloat = 56
    var cornerRadius: CGFloat = 6

    var body: some View {
        Group {
            if let url, let image = NSImage(contentsOf: url) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                ZStack {
                    Rectangle().fill(.quaternary)
                    Image(systemName: "film")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(width: size * 16 / 9, height: size)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
        .overlay(RoundedRectangle(cornerRadius: cornerRadius).strokeBorder(.separator))
    }
}

/// A recording row used by the library and the dataset pickers.
struct RecordingRow: View {
    let recording: LibraryRecording
    let thumbnailURL: URL?
    var showsFolder = false
    var folderName: String?

    var body: some View {
        HStack(spacing: 10) {
            ThumbnailView(url: thumbnailURL, size: 42)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(recording.displayName)
                        .font(.callout.weight(.medium))
                        .lineLimit(1)
                    if recording.usability == .unusable {
                        Badge(text: "Unusable", tint: .red, systemImage: "exclamationmark.triangle.fill")
                    }
                    if !recording.warningMessages.isEmpty, recording.usability == .usable {
                        Badge(text: "Warnings", tint: .orange, systemImage: "exclamationmark.circle")
                    }
                }
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 8)
            if showsFolder, let folderName {
                Badge(text: folderName, tint: .secondary)
            }
        }
        .padding(.vertical, 3)
    }

    private var subtitle: String {
        var parts: [String] = []
        if recording.pixelWidth > 0, recording.pixelHeight > 0 {
            parts.append("\(recording.pixelWidth)×\(recording.pixelHeight)")
        }
        if recording.duration > 0 { parts.append(Format.duration(recording.duration)) }
        if let fps = recording.manifest?.deliveredFPS, fps > 0 {
            parts.append("\(Format.rate(fps)) FPS")
        }
        parts.append("\(Format.integer(recording.eventCount)) events")
        return parts.joined(separator: " · ")
    }
}

struct CapabilitySummary: View {
    let capabilities: ActionCapabilities
    var emptyMessage = "No input events were recorded."

    var body: some View {
        if capabilities.isEmpty {
            Text(emptyMessage)
                .font(.callout)
                .foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(capabilities.summaryLines, id: \.self) { line in
                    HStack(alignment: .top, spacing: 6) {
                        Image(systemName: "arrow.turn.down.right")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .padding(.top, 3)
                        Text(line)
                            .font(.callout)
                            .textSelection(.enabled)
                    }
                }
            }
        }
    }
}

struct IssueList: View {
    let issues: [RecordingIssue]

    var body: some View {
        if issues.isEmpty {
            Label("No problems found.", systemImage: "checkmark.seal")
                .font(.callout)
                .foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(issues) { issue in
                    HStack(alignment: .top, spacing: 6) {
                        Image(systemName: issue.severity == .error ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                            .foregroundStyle(issue.severity == .error ? .red : .orange)
                        Text(issue.message)
                            .font(.callout)
                            .textSelection(.enabled)
                    }
                }
            }
        }
    }
}
