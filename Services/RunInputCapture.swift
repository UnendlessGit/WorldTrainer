import AppKit
import SwiftUI

/// Captures keyboard and mouse input over the Run canvas.
///
/// Events are collected by a dedicated `NSView` so capture is scoped to the
/// generated view: input starts when the view takes focus and everything is
/// released the moment it loses focus or its window stops being key. Key events
/// without the Command modifier are consumed so they never reach the rest of the
/// app, while Command shortcuts keep working.
@MainActor
final class RunInputCapture: ObservableObject {
    let box = ControlStateBox()

    @Published private(set) var isFocused = false
    @Published private(set) var state = ControlState.empty
    @Published private(set) var eventCount = 0
    @Published private(set) var message = "Click the view to capture input."

    private var refreshTimer: Timer?

    func beginReporting() {
        guard refreshTimer == nil else { return }
        let timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
        refreshTimer = timer
    }

    func endReporting() {
        refreshTimer?.invalidate()
        refreshTimer = nil
    }

    private func refresh() {
        state = box.snapshot()
        message = isFocused ? "Capturing input" : "Click the view to capture input."
    }

    func setFocused(_ focused: Bool) {
        isFocused = focused
        if focused {
            message = "Capturing input"
        } else {
            releaseAll()
            message = "Input released"
        }
    }

    /// Zeroes every held control. Called on focus loss, on stop, and whenever the
    /// Run tab is left so nothing stays stuck down.
    func releaseAll() {
        box.mutate { state in
            state.heldKeys.removeAll()
            state.pressedKeys.removeAll()
            state.heldButtons.removeAll()
            state.pressedButtons.removeAll()
            state.mouseDeltaX = 0
            state.mouseDeltaY = 0
            state.scrollX = 0
            state.scrollY = 0
            state.modifiers = 0
        }
        state = box.snapshot()
    }

    func noteEvent() {
        eventCount += 1
    }
}

/// The AppKit view that receives game input. Keeping this as a real view means
/// capture follows first-responder rules instead of hijacking global events.
final class InputCaptureView: NSView {
    weak var owner: RunInputCapture?
    private var trackingArea: NSTrackingArea?
    private var windowObserver: NSObjectProtocol?
    private var state = ControlState.empty

    override var acceptsFirstResponder: Bool { true }
    override var canBecomeKeyView: Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if let windowObserver {
            NotificationCenter.default.removeObserver(windowObserver)
            self.windowObserver = nil
        }
        guard let window else { return }
        window.acceptsMouseMovedEvents = true
        windowObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.didResignKeyNotification, object: window, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.window?.makeFirstResponder(nil)
                self.owner?.setFocused(false)
            }
        }
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let trackingArea { removeTrackingArea(trackingArea) }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
            owner: self, userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    override func becomeFirstResponder() -> Bool {
        owner?.setFocused(true)
        return true
    }

    override func resignFirstResponder() -> Bool {
        owner?.setFocused(false)
        return true
    }

    private func focus() {
        if window?.firstResponder !== self {
            window?.makeFirstResponder(self)
        }
    }

    private func flush() {
        owner?.box.update(state)
        owner?.noteEvent()
    }

    private func updatePointer(_ event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        state.pointerX = min(1, max(0, Double(point.x / max(1, bounds.width))))
        // AppKit view coordinates already have their origin at the bottom left;
        // the model expects a top-left origin like the recording region.
        state.pointerY = min(1, max(0, Double(1 - point.y / max(1, bounds.height))))
        state.pointerValid = true
    }

    private func updateModifiers(_ event: NSEvent) {
        var flags: UInt64 = 0
        if event.modifierFlags.contains(.shift) { flags |= 0x0002_0000 }
        if event.modifierFlags.contains(.control) { flags |= 0x0004_0000 }
        if event.modifierFlags.contains(.option) { flags |= 0x0008_0000 }
        if event.modifierFlags.contains(.command) { flags |= 0x0010_0000 }
        state.modifiers = flags
    }

    override func mouseDown(with event: NSEvent) {
        focus()
        button(0, down: true, event: event)
    }
    override func mouseUp(with event: NSEvent) {
        button(0, down: false, event: event)
    }
    override func rightMouseDown(with event: NSEvent) {
        focus()
        button(1, down: true, event: event)
    }
    override func rightMouseUp(with event: NSEvent) {
        button(1, down: false, event: event)
    }
    override func otherMouseDown(with event: NSEvent) {
        focus()
        button(UInt8(clamping: event.buttonNumber), down: true, event: event)
    }
    override func otherMouseUp(with event: NSEvent) {
        button(UInt8(clamping: event.buttonNumber), down: false, event: event)
    }

    private func button(_ index: UInt8, down: Bool, event: NSEvent) {
        updateModifiers(event)
        updatePointer(event)
        if index < ActionLayout.buttonCount {
            if down {
                state.heldButtons.insert(index)
                state.pressedButtons.insert(index)
            } else {
                state.heldButtons.remove(index)
            }
        }
        flush()
    }

    override func mouseMoved(with event: NSEvent) {
        move(event)
    }
    override func mouseDragged(with event: NSEvent) {
        move(event)
    }
    override func rightMouseDragged(with event: NSEvent) {
        move(event)
    }
    override func otherMouseDragged(with event: NSEvent) {
        move(event)
    }

    private func move(_ event: NSEvent) {
        updatePointer(event)
        state.mouseDeltaX += Double(event.deltaX)
        state.mouseDeltaY += Double(event.deltaY)
        flush()
    }

    override func scrollWheel(with event: NSEvent) {
        updatePointer(event)
        state.scrollX += Double(event.scrollingDeltaX)
        state.scrollY += Double(event.scrollingDeltaY)
        flush()
    }

    override func keyDown(with event: NSEvent) {
        updateModifiers(event)
        // Command chords belong to the app menu, everything else is game input.
        if event.modifierFlags.contains(.command) {
            super.keyDown(with: event)
            return
        }
        if event.isARepeat {
            flush()
            return
        }
        let code = event.keyCode
        state.heldKeys.insert(code)
        state.pressedKeys.insert(code)
        flush()
    }

    override func keyUp(with event: NSEvent) {
        updateModifiers(event)
        if event.modifierFlags.contains(.command) {
            super.keyUp(with: event)
            return
        }
        state.heldKeys.remove(event.keyCode)
        flush()
    }

    override func flagsChanged(with event: NSEvent) {
        updateModifiers(event)
        let code = event.keyCode
        if state.heldKeys.contains(code) {
            state.heldKeys.remove(code)
        } else {
            state.heldKeys.insert(code)
            state.pressedKeys.insert(code)
        }
        flush()
    }
}

struct InputCaptureLayer: NSViewRepresentable {
    let capture: RunInputCapture

    func makeNSView(context: Context) -> InputCaptureView {
        let view = InputCaptureView()
        view.owner = capture
        return view
    }

    func updateNSView(_ nsView: InputCaptureView, context: Context) {
        nsView.owner = capture
    }
}
