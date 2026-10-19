import Foundation

struct CodableRect: Codable, Hashable, Sendable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double

    static let zero = CodableRect(x: 0, y: 0, width: 0, height: 0)
}

struct RecordingVisualSpec: Codable, Hashable, Sendable {
    var width: Int?
    var height: Int?
    var colorBits: Int?
    var compression: Double?
    var locksAspectRatio: Bool?
}

struct CaptureSpec: Codable, Hashable, Sendable {
    var displayID: UInt32?
    var kind: String?
    var requestedFPS: Double?
    var showsCursor: Bool?
    var recordingVisuals: RecordingVisualSpec?
}

/// A decoded recording package manifest. Unknown JSON fields are ignored so
/// future recording schema additions remain readable.
struct RecordingManifest: Codable, Hashable, Identifiable, Sendable {
    var schemaVersion: Int = 1
    var id: UUID
    var name: String
    var createdAt: Date
    var hostStartNanos: UInt64
    var duration: Double
    var capture: CaptureSpec?
    var globalRect: CodableRect?
    var pixelWidth: Int
    var pixelHeight: Int
    var deliveredFPS: Double
    var frameCount: Int?
    var droppedFrameCount: Int?
    var eventCount: Int
    var videoFile: String = "capture.mov"
    var eventFile: String = "events.atrevents"
    var trimStart: Double = 0
    var trimEnd: Double?
    var folderID: UUID?
    var thumbnailFile: String?
    var excludedKeyCodes: Set<UInt16>?

    var requestedFPS: Double { capture?.requestedFPS ?? deliveredFPS }
    var effectiveTrimEnd: Double { trimEnd ?? duration }
    var effectiveTrimStart: Double { max(0, trimStart) }

    var encodedFrameCount: Int64 {
        if let frameCount { return Int64(max(0, frameCount)) }
        let estimate = duration * deliveredFPS
        guard estimate.isFinite, estimate > 0 else { return 0 }
        return Int64(min(Double(Int64.max), estimate.rounded()))
    }

    var isStructurallyValid: Bool {
        let end = effectiveTrimEnd
        let start = effectiveTrimStart
        let names = [videoFile, eventFile] + (thumbnailFile.map { [$0] } ?? [])
        return (1...4).contains(schemaVersion)
            && !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && duration.isFinite && duration >= 0
            && hostStartNanos > 0
            && pixelWidth > 0 && pixelHeight > 0 && pixelWidth <= 32_768 && pixelHeight <= 32_768
            && deliveredFPS.isFinite && deliveredFPS >= 0 && deliveredFPS <= 1_000
            && eventCount >= 0
            && start.isFinite && end.isFinite
            && start >= 0 && end >= start && end <= duration + 0.000_001
            && names.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("/") && !$0.contains("\0") }
    }
}

enum RecordingIssueSeverity: String, Codable, Hashable, Sendable {
    case warning
    case error
}

struct RecordingIssue: Codable, Hashable, Identifiable, Sendable {
    var id: String { "\(code):\(message)" }
    var code: String
    var message: String
    var severity: RecordingIssueSeverity

    static func error(_ code: String, _ message: String) -> RecordingIssue {
        RecordingIssue(code: code, message: message, severity: .error)
    }

    static func warning(_ code: String, _ message: String) -> RecordingIssue {
        RecordingIssue(code: code, message: message, severity: .warning)
    }
}

/// What the recording can teach the model. Run only enables these channels.
struct ActionCapabilities: Codable, Hashable, Sendable {
    var keyCodes: Set<UInt16> = []
    var buttons: Set<UInt8> = []
    var hasRelativeMouse = false
    var hasAbsolutePointer = false
    var hasScroll = false

    var isEmpty: Bool {
        keyCodes.isEmpty && buttons.isEmpty && !hasRelativeMouse && !hasAbsolutePointer && !hasScroll
    }

    var summaryLines: [String] {
        var lines: [String] = []
        if hasRelativeMouse { lines.append("Relative mouse") }
        if hasAbsolutePointer { lines.append("Absolute pointer") }
        if hasScroll { lines.append("Scroll wheel") }
        if !buttons.isEmpty {
            lines.append(buttons.sorted().map(MouseButtonNames.name(for:)).joined(separator: ", "))
        }
        if !keyCodes.isEmpty {
            lines.append(VirtualKeyNames.sortedNames(keyCodes).joined(separator: ", "))
        }
        return lines
    }

    static func union(_ values: [ActionCapabilities]) -> ActionCapabilities {
        var result = ActionCapabilities()
        for value in values {
            result.keyCodes.formUnion(value.keyCodes)
            result.buttons.formUnion(value.buttons)
            result.hasRelativeMouse = result.hasRelativeMouse || value.hasRelativeMouse
            result.hasAbsolutePointer = result.hasAbsolutePointer || value.hasAbsolutePointer
            result.hasScroll = result.hasScroll || value.hasScroll
        }
        return result
    }
}

/// A compact summary of an event stream, computed once during import.
struct EventSummary: Codable, Hashable, Sendable {
    var eventCount = 0
    var keyEventCount = 0
    var mouseMoveCount = 0
    var mouseButtonCount = 0
    var scrollCount = 0
    var flagsEventCount = 0
    var keyCodes: Set<UInt16> = []
    var buttons: Set<UInt8> = []
    var hasRelativeMouse = false
    var hasAbsolutePointer = false
    var hasScroll = false
    var firstTimestampNanos: UInt64?
    var lastTimestampNanos: UInt64?

    var capabilities: ActionCapabilities {
        ActionCapabilities(
            keyCodes: keyCodes,
            buttons: buttons,
            hasRelativeMouse: hasRelativeMouse,
            hasAbsolutePointer: hasAbsolutePointer,
            hasScroll: hasScroll
        )
    }
}
