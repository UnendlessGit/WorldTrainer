# WorldTrainer

A native macOS app for turning **recorded gameplay** into a **local, action-conditioned world model** that can be played back as a live, generated environment.

WorldTrainer imports recording packages (video + input events), trains a recurrent next-frame world model entirely on your own machine with MLX on Apple Silicon, and then *generates* new frames in real time from keyboard and mouse input.

It does not record. It has no recorder, no capture controls, and no screen recording permission requirements. It consumes recordings produced elsewhere.

---

## Contents

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Building](#building)
- [Workflow](#workflow)
- [Recording package format](#recording-package-format)
- [How the model works](#how-the-model-works)
- [Performance and tradeoffs](#performance-and-tradeoffs)
- [Storage layout](#storage-layout)
- [Verification mode](#verification-mode)
- [Project layout](#project-layout)

---

## What it does

**Library** — Import recording packages, organise them into folders, browse, preview, inspect metadata and control capabilities, and see at a glance which recordings are unusable and why. Packages are copied into the library, so originals can be moved or deleted.

**World Models** — Create any number of independently configured models. Each model owns its architecture (resolution, channel widths, sequence length), its training settings (learning rate, batch size, epochs, loss weights, seed) and its dataset selection. Every field genuinely changes what is trained. Each model also keeps its own checkpoints.

**Training** — Real, local, action-conditioned training with live loss, validation loss, gradient norm, throughput and memory readouts. Training can be paused, resumed, or stopped at any time; each of those writes a checkpoint containing weights, optimizer state and the exact dataset cursor, so a resumed run continues where it stopped rather than restarting.

**Run** — Generates a continuous stream of new frames from the trained model, driven by your keyboard and mouse. Keys, chords, modifiers, mouse buttons, relative mouse movement, absolute pointer position and scrolling are all captured, limited to the channels the model's recordings actually demonstrated. There is no maximum generation length: the recurrent memory persists for the whole session.

**Settings** — Appearance, storage location, storage usage breakdown, cache clearing, MLX memory and cache limits, default generation rate, warm-up length, and a diagnostics panel.

## Requirements

- Apple Silicon Mac (developed and benchmarked on an M3 Max with 36 GB)
- macOS 15 or later
- Xcode 16 or later (the Metal shaders MLX ships cannot be built by the plain SwiftPM command line tools)

## Building

Install [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`) or drop its binary in `.tools/xcodegen`, then:

```sh
# Generate the Xcode project (XcodeGen spec lives in project.yml)
xcodegen generate --spec project.yml

xcodebuild -project WorldTrainer.xcodeproj \
  -scheme WorldTrainer \
  -configuration Release \
  -derivedDataPath .build \
  -destination 'platform=macOS,arch=arm64' \
  build
```

The generated `WorldTrainer.xcodeproj` is committed, so you can also open it directly in Xcode. Regenerate it after adding or removing source files.

The built app lands in `.build/Build/Products/Release/WorldTrainer.app`.

`project.yml` pins [mlx-swift](https://github.com/ml-explore/mlx-swift) to `0.32.3`; Xcode resolves and builds it on first build.

## Workflow

1. **Library → Import Recordings…** Choose a folder containing one or more recording packages (or a single package folder). Each package is inspected, validated, thumbnailed and copied into the library.
2. **World Models → +** Name the model and pick a starting configuration. Presets cover fast iteration, the balanced default, and high detail. Choose which recordings it learns from — only those are used.
3. **Training → Start Training.** Watch the loss fall, pause or stop at any point. Resume continues from the newest checkpoint that matches the configuration and dataset.
4. **Run → Start Generation.** The app loads the checkpoint, decodes a short warm-up window from a recording to fill the model's memory, and then generates every frame from that point on. Click the view to capture input.

A model must be trained before it can be run. The Run tab lists only trained models and shows which checkpoint each session uses.

## Recording package format

A recording package is a folder containing:

| File | Purpose |
| --- | --- |
| `manifest.json` | Recording metadata: id, name, timestamps, duration, video geometry, delivered FPS, frame and event counts, trim range, capture spec, capture region |
| `events.atrevents` | Binary input event stream |
| `capture.mov` | Recorded video |
| `thumbnail.jpg` | Optional preview image; generated on import when missing |

### `events.atrevents`

A 12-byte header (`ATREVT01` followed by a little-endian `UInt32` version, currently `1`), then a sequence of fixed **72-byte** records:

| Offset | Type | Field |
| --- | --- | --- |
| 0 | `UInt64` | host timestamp in nanoseconds |
| 8 | `UInt8` | kind: `1` mouseMove, `2` mouseButton, `3` scroll, `4` key, `5` flags |
| 9 | `UInt8` | isDown |
| 10 | `UInt8` | button |
| 11 | `UInt8` | padding |
| 12 | `UInt16` | keyCode |
| 14 | `UInt16` | padding |
| 16 | `UInt64` | modifier flags |
| 24 | `Double` | x |
| 32 | `Double` | y |
| 40 | `Double` | deltaX |
| 48 | `Double` | deltaY |
| 56 | `Double` | scrollX |
| 64 | `Double` | scrollY |

Records are ordered by timestamp. `manifest.hostStartNanos` is the host clock value at recording offset zero, so `trimStart`/`trimEnd` map onto the event stream by simple addition.

### Control vector

Events are accumulated into one **146-value** control row per model frame:

| Range | Meaning |
| --- | --- |
| `0..<2` | absolute pointer position, normalised to the captured region |
| `2..<4` | relative mouse movement, scaled and clamped |
| `4..<12` | mouse buttons (held or pressed during the interval) |
| `12..<14` | scroll deltas |
| `14..<142` | keyboard key codes (held or pressed during the interval) |
| `142..<146` | Shift, Control, Option, Command |

Modifier key codes are excluded from the key block and represented only in the modifier block, so a held Shift is not double-counted.

## How the model works

The world model is a compact, action-conditioned recurrent convolutional network:

```
frame[t] ──▶ encoder (3 conv blocks, /8) ──▶ ┌──────────────────┐
                                             │  ConvGRU cell    │──▶ hidden state (persistent)
action[t] ─▶ MLP embedding ──── broadcast ──▶└──────────────────┘
                                                        │
                          skip connections ─────────────┼──▶ decoder (3 up blocks)
                                                        │
                             frame[t] + tanh(delta) ◀────┘──▶ frame[t+1]
```

- The encoder downsamples by 8 while widening channels.
- A convolutional GRU carries a persistent spatial hidden state. It is never reset on a timer, so long sessions keep their context.
- The action row is embedded and injected both into the recurrent cell and into every decoder block, so control affects not only *what moves* but *what is rendered*.
- The decoder predicts a bounded residual on top of the current frame (`tanh` scaled), which keeps static scenes stable instead of drifting.
- Training uses teacher forcing over `sequenceLength` frame steps; generation is autoregressive over the model's own output.

Loss is L1 pixel error plus optional temporal-consistency and edge terms, weighted per model configuration.

Optimizer state is a hand-written AdamW with explicitly serialisable moments, which is what makes exact resume possible.

## Performance and tradeoffs

Generation speed depends almost entirely on resolution and channel widths. Measured with the Release build on an M3 Max (36 GB), machine otherwise idle, at the default channel widths, per generated frame:

| Resolution | Measured generation | Notes |
| --- | --- | --- |
| 128 × 80 | ≈ 630–670 FPS (1.5 ms/frame) | Coarsest picture, easiest to hit high frame rates |
| 160 × 104 | ≈ 530–560 FPS (1.8 ms/frame) | The default; the intended balance for 15–30 FPS |
| 192 × 120 | ≈ 480 FPS (2.1 ms/frame) | Sharpest output, heaviest training and generation |

A full trained-model rollout in the self-test measured ≈ 670–725 FPS (1.4–1.5 ms/frame) end to end.

These are compute figures — how fast the model can produce frames — not the rate you watch. The Run tab paces generation to its target rate (5–120 FPS) and reports the displayed rate and the remaining headroom separately. A Debug build is roughly four times slower than Release, so build Release for real use.

Numbers move with machine state: the same binary measured 234 FPS (4.3 ms/frame) immediately after a heavy compile and 670–725 FPS on an idle machine. Run the verification mode below to measure your own machine; the report includes a per-resolution benchmark and the measured generation rate of a trained model.

Quality tradeoffs are real and worth stating plainly:

- Generated frames are **not** the recorded footage. They are the model's own predictions, so they are softer than the source video and can drift or smear over long sessions.
- Short training runs on small datasets produce recognisable motion and stable static regions but poor fine detail. More epochs, higher resolution and wider channels all help, at a direct cost in speed.
- The model learns only the games it was shown. Control channels that never appeared in the selected recordings are disabled in Run rather than guessed.
- The first frames after warm-up are the most faithful; coherence degrades as the rollout moves away from the training distribution.

## Storage layout

Everything lives under one data root, configurable in Settings:

```
<root>/
  library.json                          # library index
  Recordings/<recording id>/            # imported packages, copied verbatim
  models.json                           # model index
  Models/<model id>/model.json          # per-model configuration
  Models/<model id>/checkpoints/<id>/   # weights.safetensors, optimizer.safetensors, manifest.json
  TrainingCache/<model id>/<fingerprint>/
```

The dataset cache is a pair of flat binary files (`frames.bin`, `actions.bin`) plus a manifest, keyed by a fingerprint of the resolution, FPS, architecture and recording identities. Changing any of those rebuilds the cache automatically.

## Verification mode

The app can verify the entire pipeline headlessly, without the interface:

```sh
.build/Build/Products/Debug/WorldTrainer.app/Contents/MacOS/WorldTrainer \
  --self-test --self-test-package ./RecordingFormat
```

It imports the package, builds the dataset cache, trains a real model with a deliberate pause and resume, checks that weights actually changed, generates frames from the trained checkpoint, measures generation speed, and benchmarks all three resolutions. It prints a JSON report and exits non-zero on failure.

This is a genuine end-to-end run of the same code the interface drives, not a mock.

## Project layout

```
App/         Application entry point and tab shell
Core/        Recording format, action layout, dataset building, video decoding, filesystem layout
Learning/    World model, optimizer, training worker, checkpoint store, generation engine
Services/    Input capture, generation session/coordinator, self test
Stores/      Settings, library, models and the shared app store
Views/       The five tabs and shared components
Resources/   Info.plist and asset catalog
```
