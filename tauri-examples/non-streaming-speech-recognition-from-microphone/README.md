# Non-Streaming Speech Recognition from Microphone

A [Tauri v2](https://v2.tauri.app/) desktop app that captures microphone audio
and transcribes it in using offline ASR with
[Silero VAD](https://github.com/snakers4/silero-vad).

Pre-built apps (macOS, Linux, Windows) are available at:

> <https://k2-fsa.github.io/sherpa/onnx/tauri/pre-built-app.html#non-streaming-speech-recognition-from-microphone>

## Features

- **62+ ASR models** — SenseVoice, Paraformer, Whisper, Transducer, Moonshine, and more
- **Live microphone capture** via [cpal](https://github.com/RustAudioGroup/cpal) — works on Linux, macOS, and Windows
- **VAD-triggered recognition** — detects when you stop speaking and transcribes each segment
- **Wall-clock timestamps** — results show real times (e.g., "2026-04-21 18:52:28")
- **Click-to-seek** — click any row in the results table to jump to that segment in the recording
- **Recording playback** — play back the full recording after stopping
- **Segment WAV export** — save individual speech segments as WAV files
- **Full recording export** — save the entire recording as a WAV file
- **SRT subtitle export** — save results as `.srt` files
- **Copy results** — copy plain text or timestamped text to clipboard
- **Configurable VAD parameters** — adjust threshold, silence/speech duration, and thread count at runtime
- **Cross-platform** — macOS (universal), Linux (x64/aarch64), Windows (x64)
- **Pure Rust audio capture** via cpal — no system dependencies beyond platform audio APIs

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable)
- [Node.js](https://nodejs.org/) (for the Tauri CLI)
- [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/) (platform-specific system dependencies)
- **Linux only**: ALSA development headers (`libasound2-dev` on Debian/Ubuntu)

Install npm dependencies:

```bash
npm install
```

## Quick Start

This example bundles the **SenseVoice int8** model (model type 15), which
supports Chinese, English, Japanese, Korean, and Cantonese.

### 1. Download the model and Silero VAD

```bash
cd tauri-examples/non-streaming-speech-recognition-from-microphone

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

### 2. Copy assets into src-tauri

Tauri bundles files from `src-tauri/assets/`. Place the model directory
(keeping its original name) and `silero_vad.onnx` inside it:

```bash
mkdir -p src-tauri/assets
cp -a sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17 src-tauri/assets/
cp silero_vad.onnx src-tauri/assets/
```

Expected layout:

```
src-tauri/assets/
├── silero_vad.onnx
└── sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/
    ├── model.int8.onnx
    └── tokens.txt
```

### 3. Run in development mode

```bash
npm run dev
```

This opens the app window. Click **Start Recording**, speak into your
microphone, and recognized segments appear after you stop speaking. Click
**Stop Recording** to finish — you can then play back the recording and
export results.

### 4. Build a release binary

```bash
npm run build
```

The output is in `src-tauri/target/release/bundle/`.

## Using a Different Model

The app uses two constants in [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs) to select which model
to bundle:

```rust
const MODEL_TYPE: u32 = 15;
const MODEL_NAME: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17";
```

To switch models:

1. Pick a model type from [`src-tauri/src/model_registry.rs`](src-tauri/src/model_registry.rs) (types 0--61).
2. Update `MODEL_TYPE` and `MODEL_NAME` in [`src-tauri/src/lib.rs`](src-tauri/src/lib.rs).
3. Download the corresponding model and place its directory into
   `src-tauri/assets/`, keeping the original directory name.
4. Run `npm run dev` to test.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Frontend (HTML + JS)                            │
│  index.html / main.js / styles.css               │
│                                                  │
│  invoke("start_recording")                       │
│  invoke("stop_recording")                        │
│  setInterval → invoke("get_recording_state")     │
└──────────────┬───────────────────────────────────┘
               │ Tauri IPC (invoke)
┌──────────────▼───────────────────────────────────┐
│  Backend (Rust)                                  │
│                                                  │
│  lib.rs                                          │
│  ├── start_recording() → spawns recording thread │
│  ├── run_recording()   → cpal → VAD → ASR        │
│  ├── stop_recording()  → signals thread to stop  │
│  ├── get_recording_state() → poll results        │
│  ├── save_segment_as_wav()                       │
│  ├── save_all_audio()                            │
│  ├── get_recorded_audio_path()                   │
│  └── export_srt()                                │
│                                                  │
│  model_registry.rs (auto-generated, 62 models)   │
└──────────────────────────────────────────────────┘
```

**Processing pipeline:**

1. **cpal** captures microphone audio (any native sample rate/format).
2. Audio is downmixed to mono f32 and sent through an mpsc channel.
3. **LinearResampler** resamples to 16 kHz if the native sample rate differs.
4. **Silero VAD** processes 512-sample (32 ms) windows and detects speech segments.
5. **OfflineRecognizer** transcribes each completed speech segment.
6. Results (with wall-clock timestamps) are accumulated and polled by the frontend every 200 ms.
7. All captured audio is stored for playback and export.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Models are still loading" on startup | Models load in a background thread. Wait a few seconds. Large models take longer. |
| "Failed to create recognizer" | Check that `src-tauri/assets/<MODEL_NAME>/` exists and contains the expected `.onnx` and `tokens.txt` files. |
| "No default input device" | Ensure a microphone is connected and recognized by your OS. |
| No audio captured | On macOS, grant microphone permission when prompted. On Linux, ensure ALSA/PulseAudio is working. |
| App crashes on startup | Run `npm run dev` and check stderr for `[init]` or `[build_models]` log lines. |

## How It Works

1. Click **Start Recording** to begin capturing microphone audio.
2. Audio is captured by cpal, downmixed to mono, and resampled to 16 kHz.
3. Audio is fed to the VAD in 512-sample (32 ms) chunks.
4. Each detected speech segment is decoded by the offline recognizer.
5. Results are displayed in a table with wall-clock timestamps and text.
6. Click **Stop Recording** to stop capture and flush remaining audio.
7. Play back the recording or export results as SRT/WAV.
