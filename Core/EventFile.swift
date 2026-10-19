import Foundation

enum InputEventKind: UInt8, Codable, Sendable, CaseIterable {
    case mouseMove = 1
    case mouseButton = 2
    case scroll = 3
    case key = 4
    case flags = 5
}

struct RecordingEvent: Hashable, Sendable {
    var timestampNanos: UInt64
    var kind: InputEventKind
    var x: Double = 0
    var y: Double = 0
    var deltaX: Double = 0
    var deltaY: Double = 0
    var button: UInt8 = 0
    var scrollX: Double = 0
    var scrollY: Double = 0
    var keyCode: UInt16 = 0
    var modifiers: UInt64 = 0
    var isDown = false
}

enum EventFileError: LocalizedError, Sendable {
    case unreadable(String)
    case badMagic
    case unsupportedVersion(UInt32)
    case truncated
    case unknownKind(UInt8)
    case unordered(Int)
    case nonFinite(Int)
    case countMismatch(expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .unreadable(let detail): return "The input event file could not be read: \(detail)"
        case .badMagic: return "The input event file is not a valid ATREVT01 package."
        case .unsupportedVersion(let version): return "The input event file uses unsupported version \(version)."
        case .truncated: return "The input event file is truncated."
        case .unknownKind(let kind): return "The input event file contains unknown event kind \(kind)."
        case .unordered(let index): return "The input event file is not ordered by capture time at record \(index)."
        case .nonFinite(let index): return "The input event file contains a non-finite control value at record \(index)."
        case .countMismatch(let expected, let actual): return "The manifest reports \(expected) events but the file contains \(actual)."
        }
    }
}

/// Reader for the fixed-width ATREVT01 input event stream.
///
/// Layout: an 8-byte magic ("ATREVT01"), a little-endian UInt32 version, then
/// 72-byte records:
/// `timestamp UInt64 | kind UInt8 | isDown UInt8 | button UInt8 | pad UInt8 |
///  keyCode UInt16 | pad UInt16 | modifiers UInt64 | x f64 | y f64 |
///  deltaX f64 | deltaY f64 | scrollX f64 | scrollY f64`.
enum EventFile {
    static let recordSize = 72

    static func recordCount(in data: Data) throws -> Int {
        guard data.count >= 12 else { throw EventFileError.truncated }
        guard String(data: data.prefix(8), encoding: .utf8) == "ATREVT01" else {
            throw EventFileError.badMagic
        }
        let version = readUInt32(data, at: 8)
        guard version == 1 else { throw EventFileError.unsupportedVersion(version) }
        let payload = data.count - 12
        guard payload % recordSize == 0 else { throw EventFileError.truncated }
        return payload / recordSize
    }

    static func read(url: URL) throws -> [RecordingEvent] {
        let data = try mappedData(url)
        let count = try recordCount(in: data)
        var result: [RecordingEvent] = []
        result.reserveCapacity(count)
        try forEach(in: data, count: count) { result.append($0) }
        return result
    }

    @discardableResult
    static func forEachEvent(url: URL, _ body: (RecordingEvent) throws -> Void) throws -> Int {
        let data = try mappedData(url)
        let count = try recordCount(in: data)
        try forEach(in: data, count: count, body)
        return count
    }

    static func validate(url: URL, expectedCount: Int?) throws -> Int {
        let count = try forEachEvent(url: url) { _ in }
        if let expectedCount, expectedCount != count {
            throw EventFileError.countMismatch(expected: expectedCount, actual: count)
        }
        return count
    }

    static func summarize(url: URL, expectedCount: Int?) throws -> EventSummary {
        let data = try mappedData(url)
        let count = try recordCount(in: data)
        if let expectedCount, expectedCount != count {
            throw EventFileError.countMismatch(expected: expectedCount, actual: count)
        }
        var summary = EventSummary(eventCount: count)
        var sawMouseMove = false
        var staticPointer = true
        var firstPointer: (Double, Double)?
        try forEach(in: data, count: count) { event in
            if summary.firstTimestampNanos == nil { summary.firstTimestampNanos = event.timestampNanos }
            summary.lastTimestampNanos = event.timestampNanos
            switch event.kind {
            case .key:
                summary.keyEventCount += 1
                if event.isDown { summary.keyCodes.insert(event.keyCode) }
            case .flags:
                summary.flagsEventCount += 1
                summary.keyCodes.formUnion(modifierKeyCodes(in: event.modifiers))
            case .mouseMove:
                summary.mouseMoveCount += 1
                sawMouseMove = true
                if abs(event.deltaX) > 0.001 || abs(event.deltaY) > 0.001 {
                    summary.hasRelativeMouse = true
                }
                if let firstPointer {
                    if abs(event.x - firstPointer.0) > 0.5 || abs(event.y - firstPointer.1) > 0.5 {
                        staticPointer = false
                    }
                } else {
                    firstPointer = (event.x, event.y)
                }
            case .mouseButton:
                summary.mouseButtonCount += 1
                summary.buttons.insert(event.button)
            case .scroll:
                summary.scrollCount += 1
                if abs(event.scrollX) > 0.001 || abs(event.scrollY) > 0.001 {
                    summary.hasScroll = true
                }
            }
        }
        summary.hasAbsolutePointer = sawMouseMove && !staticPointer
        return summary
    }

    /// Incremental cursor used by the dataset builder so long recordings never
    /// require materializing every event in memory.
    struct EventCursor {
        private let data: Data
        let count: Int
        private(set) var index = 0

        init(url: URL) throws {
            let mapped = try mappedData(url)
            data = mapped
            count = try recordCount(in: mapped)
        }

        var isAtEnd: Bool { index >= count }

        mutating func peek() -> RecordingEvent? {
            guard index < count else { return nil }
            return EventFile.decode(data, index: index)
        }

        mutating func next() -> RecordingEvent? {
            let value = peek()
            if value != nil { index += 1 }
            return value
        }
    }

    static func decode(_ data: Data, index: Int) -> RecordingEvent? {
        let offset = 12 + index * recordSize
        guard offset + recordSize <= data.count else { return nil }
        return data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            let pointer = base.advanced(by: offset)
            let kindRaw = pointer.loadUnaligned(fromByteOffset: 8, as: UInt8.self)
            guard let kind = InputEventKind(rawValue: kindRaw) else { return nil }
            return RecordingEvent(
                timestampNanos: pointer.loadUnaligned(fromByteOffset: 0, as: UInt64.self),
                kind: kind,
                x: pointer.loadUnaligned(fromByteOffset: 24, as: Double.self),
                y: pointer.loadUnaligned(fromByteOffset: 32, as: Double.self),
                deltaX: pointer.loadUnaligned(fromByteOffset: 40, as: Double.self),
                deltaY: pointer.loadUnaligned(fromByteOffset: 48, as: Double.self),
                button: pointer.loadUnaligned(fromByteOffset: 10, as: UInt8.self),
                scrollX: pointer.loadUnaligned(fromByteOffset: 56, as: Double.self),
                scrollY: pointer.loadUnaligned(fromByteOffset: 64, as: Double.self),
                keyCode: pointer.loadUnaligned(fromByteOffset: 12, as: UInt16.self),
                modifiers: pointer.loadUnaligned(fromByteOffset: 16, as: UInt64.self),
                isDown: pointer.loadUnaligned(fromByteOffset: 9, as: UInt8.self) != 0)
        }
    }

    private static func modifierKeyCodes(in flags: UInt64) -> Set<UInt16> {
        let mappings: [(UInt64, UInt16)] = [
            (0x0002_0000, 56), // Shift
            (0x0004_0000, 59), // Control
            (0x0008_0000, 58), // Option
            (0x0010_0000, 55), // Command
        ]
        return Set(mappings.compactMap { flags & $0.0 != 0 ? $0.1 : nil })
    }

    private static func mappedData(_ url: URL) throws -> Data {
        do {
            return try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw EventFileError.unreadable(error.localizedDescription)
        }
    }

    private static func forEach(in data: Data, count: Int, _ body: (RecordingEvent) throws -> Void) throws {
        var previousTimestamp: UInt64?
        try data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            for index in 0..<count {
                let offset = 12 + index * recordSize
                let pointer = base.advanced(by: offset)
                let timestamp = pointer.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
                let kindRaw = pointer.loadUnaligned(fromByteOffset: 8, as: UInt8.self)
                guard let kind = InputEventKind(rawValue: kindRaw) else {
                    throw EventFileError.unknownKind(kindRaw)
                }
                let isDown = pointer.loadUnaligned(fromByteOffset: 9, as: UInt8.self) != 0
                let button = pointer.loadUnaligned(fromByteOffset: 10, as: UInt8.self)
                let keyCode = pointer.loadUnaligned(fromByteOffset: 12, as: UInt16.self)
                let modifiers = pointer.loadUnaligned(fromByteOffset: 16, as: UInt64.self)
                let x = pointer.loadUnaligned(fromByteOffset: 24, as: Double.self)
                let y = pointer.loadUnaligned(fromByteOffset: 32, as: Double.self)
                let deltaX = pointer.loadUnaligned(fromByteOffset: 40, as: Double.self)
                let deltaY = pointer.loadUnaligned(fromByteOffset: 48, as: Double.self)
                let scrollX = pointer.loadUnaligned(fromByteOffset: 56, as: Double.self)
                let scrollY = pointer.loadUnaligned(fromByteOffset: 64, as: Double.self)
                if let previousTimestamp, timestamp < previousTimestamp {
                    throw EventFileError.unordered(index)
                }
                previousTimestamp = timestamp
                guard [x, y, deltaX, deltaY, scrollX, scrollY].allSatisfy({ $0.isFinite }) else {
                    throw EventFileError.nonFinite(index)
                }
                try body(RecordingEvent(
                    timestampNanos: timestamp, kind: kind, x: x, y: y,
                    deltaX: deltaX, deltaY: deltaY, button: button,
                    scrollX: scrollX, scrollY: scrollY, keyCode: keyCode,
                    modifiers: modifiers, isDown: isDown))
            }
        }
    }

    private static func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
    }
}

/// Accumulates raw events into one 146-value control row per frame interval.
struct ActionAccumulator {
    var pointerX = 0.0
    var pointerY = 0.0
    var pointerValid = false
    var deltaX = 0.0
    var deltaY = 0.0
    var scrollX = 0.0
    var scrollY = 0.0
    var heldKeys: Set<UInt16> = []
    var pressedKeys: Set<UInt16> = []
    var heldButtons: Set<UInt8> = []
    var pressedButtons: Set<UInt8> = []
    var modifiers: UInt64 = 0
    var pressedModifierMasks: UInt64 = 0
    var region: CodableRect

    init(region: CodableRect, initialPointer: (Double, Double)? = nil) {
        self.region = region
        if let initialPointer {
            pointerX = initialPointer.0
            pointerY = initialPointer.1
            pointerValid = true
        } else {
            pointerX = region.x + region.width / 2
            pointerY = region.y + region.height / 2
        }
    }

    mutating func consume(_ event: RecordingEvent) {
        pressedModifierMasks |= event.modifiers
        modifiers = event.modifiers
        switch event.kind {
        case .mouseMove:
            pointerX = event.x
            pointerY = event.y
            pointerValid = true
            deltaX += event.deltaX
            deltaY += event.deltaY
        case .mouseButton:
            pointerX = event.x
            pointerY = event.y
            pointerValid = true
            if event.isDown {
                heldButtons.insert(event.button)
                pressedButtons.insert(event.button)
            } else {
                heldButtons.remove(event.button)
            }
        case .scroll:
            pointerX = event.x
            pointerY = event.y
            pointerValid = true
            scrollX += event.scrollX
            scrollY += event.scrollY
        case .key:
            if event.isDown {
                heldKeys.insert(event.keyCode)
                pressedKeys.insert(event.keyCode)
            } else {
                heldKeys.remove(event.keyCode)
            }
        case .flags:
            break
        }
    }

    mutating func endTick() {
        deltaX = 0
        deltaY = 0
        scrollX = 0
        scrollY = 0
        pressedKeys.removeAll(keepingCapacity: true)
        pressedButtons.removeAll(keepingCapacity: true)
        pressedModifierMasks = 0
    }

    func actionValues() -> [Float] {
        var values = [Float](repeating: 0, count: ActionLayout.count)
        let width = max(1, region.width)
        let height = max(1, region.height)
        values[0] = Float(min(1, max(0, (pointerX - region.x) / width)))
        values[1] = Float(min(1, max(0, (pointerY - region.y) / height)))
        values[2] = ActionLayout.normalizedRelativeMouse(deltaX)
        values[3] = ActionLayout.normalizedRelativeMouse(deltaY)
        for button in heldButtons.union(pressedButtons) where button < ActionLayout.buttonCount {
            values[4 + Int(button)] = 1
        }
        values[12] = ActionLayout.normalizedScroll(scrollX)
        values[13] = ActionLayout.normalizedScroll(scrollY)
        for key in heldKeys.union(pressedKeys) where key < 128 {
            guard !ActionLayout.commandOptionControlKeyCodeSet.contains(key) else { continue }
            values[14 + Int(key)] = 1
        }
        let effective = modifiers | pressedModifierMasks
        values[142] = effective & 0x0002_0000 != 0 ? 1 : 0
        values[143] = effective & 0x0004_0000 != 0 ? 1 : 0
        values[144] = effective & 0x0008_0000 != 0 ? 1 : 0
        values[145] = effective & 0x0010_0000 != 0 ? 1 : 0
        return values
    }
}
