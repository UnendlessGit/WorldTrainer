import Foundation

/// The action vector used by the world model.
///
/// The layout matches the 146-value control row produced by the recording
/// package format so imported data can be consumed without lossy translation:
///
/// - 0..<2 absolute pointer position, normalized to the captured region
/// - 2..<4 relative mouse movement
/// - 4..<12 mouse buttons (held or pressed within the interval)
/// - 12..<14 scroll deltas
/// - 14..<142 keyboard key codes (held or pressed within the interval)
/// - 142..<146 Shift, Control, Option, Command
enum ActionLayout {
    static let absoluteMouse = 0..<2
    static let relativeMouse = 2..<4
    static let buttons = 4..<12
    static let scroll = 12..<14
    static let keyboard = 14..<142
    static let modifiers = 142..<146
    static let count = 146

    static let commandOptionControlKeyCodes: [UInt16] = [54, 55, 58, 59, 61, 62]
    static let shiftKeyCodes: Set<UInt16> = [56, 60]
    static let commandOptionControlKeyCodeSet = Set(commandOptionControlKeyCodes)

    static let buttonCount = 8

    /// Raw relative deltas are divided by this before clamping, so ordinary
    /// game-camera motion lands in a well conditioned range.
    static let relativeMouseScale: Float = 80

    static func normalizedRelativeMouse(_ value: Double) -> Float {
        min(1, max(-1, Float(value) / relativeMouseScale))
    }

    static func denormalizedRelativeMouse(_ value: Float) -> Double {
        Double(value) * Double(relativeMouseScale)
    }

    static func normalizedScroll(_ value: Double) -> Float {
        min(1, max(-1, Float(value) / 20))
    }
}

extension ActionLayout {
    /// Builds one model-ready control row from a live input snapshot.
    ///
    /// Channels the selected recordings never demonstrated are forced to zero so
    /// the model is only ever asked to continue behaviour it was trained on.
    static func values(
        for state: ControlState,
        capabilities: ActionCapabilities,
        region: CodableRect = CodableRect(x: 0, y: 0, width: 1, height: 1)
    ) -> [Float] {
        var values = [Float](repeating: 0, count: count)
        let width = max(1e-9, region.width)
        let height = max(1e-9, region.height)
        if capabilities.hasAbsolutePointer, state.pointerValid {
            values[0] = Float(min(1, max(0, (state.pointerX - region.x) / width)))
            values[1] = Float(min(1, max(0, (state.pointerY - region.y) / height)))
        }
        if capabilities.hasRelativeMouse {
            values[2] = normalizedRelativeMouse(state.mouseDeltaX)
            values[3] = normalizedRelativeMouse(state.mouseDeltaY)
        }
        if !capabilities.buttons.isEmpty {
            for button in state.heldButtons.union(state.pressedButtons)
            where button < buttonCount && capabilities.buttons.contains(button) {
                values[4 + Int(button)] = 1
            }
        }
        if capabilities.hasScroll {
            values[12] = normalizedScroll(state.scrollX)
            values[13] = normalizedScroll(state.scrollY)
        }
        if !capabilities.keyCodes.isEmpty {
            for key in state.heldKeys.union(state.pressedKeys)
            where key < 128 && capabilities.keyCodes.contains(key) {
                guard !commandOptionControlKeyCodeSet.contains(key) else { continue }
                values[14 + Int(key)] = 1
            }
        }
        values[142] = state.modifiers & 0x0002_0000 != 0 ? 1 : 0
        values[143] = state.modifiers & 0x0004_0000 != 0 ? 1 : 0
        values[144] = state.modifiers & 0x0008_0000 != 0 ? 1 : 0
        values[145] = state.modifiers & 0x0010_0000 != 0 ? 1 : 0
        return values
    }
}

/// Thread-safe hand-off of live input between the capture view (main thread)
/// and the generation queue.
final class ControlStateBox: @unchecked Sendable {
    private let lock = NSLock()
    private var state = ControlState.empty

    func update(_ new: ControlState) {
        lock.lock()
        state = new
        lock.unlock()
    }

    /// Applies a mutation to the stored state, for example releasing everything.
    func mutate(_ body: (inout ControlState) -> Void) {
        lock.lock()
        body(&state)
        lock.unlock()
    }

    /// Returns the current state and clears the fields that describe a single
    /// frame interval, exactly like the recording accumulator does.
    func consumeForTick() -> ControlState {
        lock.lock()
        var snapshot = state
        state.endTick()
        lock.unlock()
        return snapshot
    }

    func snapshot() -> ControlState {
        lock.lock()
        defer { lock.unlock() }
        return state
    }
}

/// A concrete input state captured from the live app window during Run.
struct ControlState: Equatable, Sendable {
    var pointerX: Double = 0
    var pointerY: Double = 0
    var pointerValid = false
    var mouseDeltaX: Double = 0
    var mouseDeltaY: Double = 0
    var scrollX: Double = 0
    var scrollY: Double = 0
    var heldKeys: Set<UInt16> = []
    var pressedKeys: Set<UInt16> = []
    var heldButtons: Set<UInt8> = []
    var pressedButtons: Set<UInt8> = []
    var modifiers: UInt64 = 0

    static let empty = ControlState()

    mutating func endTick() {
        mouseDeltaX = 0
        mouseDeltaY = 0
        scrollX = 0
        scrollY = 0
        pressedKeys.removeAll(keepingCapacity: true)
        pressedButtons.removeAll(keepingCapacity: true)
    }
}

enum VirtualKeyNames {
    private static let names: [UInt16: String] = [
        0: "A", 1: "S", 2: "D", 3: "F", 4: "H", 5: "G", 6: "Z", 7: "X", 8: "C", 9: "V",
        11: "B", 12: "Q", 13: "W", 14: "E", 15: "R", 16: "Y", 17: "T",
        18: "1", 19: "2", 20: "3", 21: "4", 22: "6", 23: "5", 24: "=", 25: "9",
        26: "7", 27: "-", 28: "8", 29: "0", 30: "]", 31: "O", 32: "U", 33: "[",
        34: "I", 35: "P", 36: "Return", 37: "L", 38: "J", 39: "'", 40: "K",
        41: ";", 42: "\\", 43: ",", 44: "/", 45: "N", 46: "M", 47: ".",
        48: "Tab", 49: "Space", 50: "`", 51: "Delete", 53: "Escape",
        54: "Right Command", 55: "Command", 56: "Shift", 57: "Caps Lock",
        58: "Option", 59: "Control", 60: "Right Shift", 61: "Right Option", 62: "Right Control",
        63: "Fn", 64: "F17", 65: "Keypad .", 67: "Keypad *", 69: "Keypad +",
        71: "Keypad Clear", 75: "Keypad /", 76: "Keypad Enter", 78: "Keypad -",
        79: "F18", 80: "F19", 81: "Keypad =", 82: "Keypad 0", 83: "Keypad 1",
        84: "Keypad 2", 85: "Keypad 3", 86: "Keypad 4", 87: "Keypad 5", 88: "Keypad 6",
        89: "Keypad 7", 91: "Keypad 8", 92: "Keypad 9", 96: "F5", 97: "F6",
        98: "F7", 99: "F3", 100: "F8", 101: "F9", 103: "F11", 105: "F13",
        106: "F16", 107: "F14", 109: "F10", 111: "F12", 113: "F15",
        114: "Help", 115: "Home", 116: "Page Up", 117: "Forward Delete",
        118: "F4", 119: "End", 120: "F2", 121: "Page Down", 122: "F1",
        123: "Left Arrow", 124: "Right Arrow", 125: "Down Arrow", 126: "Up Arrow",
    ]

    static func name(for keyCode: UInt16) -> String {
        if let name = names[keyCode] { return name }
        return "Key \(keyCode)"
    }

    static func sortedNames(_ codes: Set<UInt16>) -> [String] {
        codes.sorted { lhs, rhs in
            let l = names[lhs] ?? "Key \(lhs)"
            let r = names[rhs] ?? "Key \(rhs)"
            if l.count != r.count { return l.count < r.count }
            return l < r
        }.map { name(for: $0) }
    }
}

enum MouseButtonNames {
    static func name(for button: UInt8) -> String {
        switch button {
        case 0: return "Left Click"
        case 1: return "Right Click"
        case 2: return "Middle Click"
        case 3: return "Button 4"
        case 4: return "Button 5"
        default: return "Button \(Int(button) + 1)"
        }
    }
}
