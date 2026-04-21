# Non-Streaming Speech Recognition from File

A [Tauri v2](https://v2.tauri.app/) desktop app that transcribes audio and video
files using offline ASR with [Silero VAD](https://github.com/snakers4/silero-vad).

Pre-built apps (macOS, Linux, Windows) are available at:

> <https://k2-fsa.github.io/sherpa/onnx/tauri/pre-built-app.html#non-streaming-speech-recognition-from-file>

## Screenshots

|1|2|3|
|---|---|---|
|<img width="1800" height="1392" alt="Screenshot 2026-04-20 at 14 59 39" src="https://github.com/user-attachments/assets/81369094-0e8a-4be5-b15e-f83d16821be0" />|<img width="1768" height="1388" alt="Screenshot 2026-04-20 at 14 59 49" src="https://github.com/user-attachments/assets/62923691-9446-476b-838b-df3b5565d1a2" />|<img width="1798" height="1398" alt="Screenshot 2026-04-20 at 15 00 00" src="https://github.com/user-attachments/assets/a199c25e-1d1a-4907-9971-a779a0919c44" />|

## Video Demo

<https://www.bilibili.com/video/BV1cXoKBhEdz>

## Features

- **62+ ASR models** — SenseVoice, Paraformer, Whisper, Transducer, Moonshine, and more
- **Audio and video input** — MP3, FLAC, AAC, OGG, WAV, AIFF, CAF, MP4, MKV, WebM
- **Live subtitle overlay** — subtitles sync with playback in real time
- **Click-to-seek** — click any row in the results table to jump to that segment
- **SRT subtitle export** — save results as `.srt` files
- **Segment WAV export** — save individual speech segments as WAV files
- **Copy results** — copy plain text or timestamped text to clipboard
- **Progress tracking** with cancellation support
- **Cross-platform** — macOS (universal), Linux (x64/aarch64), Windows (x64)
- **Pure Rust audio decoding** via [symphonia](https://github.com/pdeljanov/Symphonia) — no system dependencies

## Supported Formats

Any format supported by symphonia:

| Type  | Formats                                          |
|-------|--------------------------------------------------|
| Audio | MP3, FLAC, AAC, OGG/Vorbis, WAV, AIFF, CAF, ADPCM |
| Video | MP4/M4A, MKV, WebM (audio track extracted)       |

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable)
- [Node.js](https://nodejs.org/) (for the Tauri CLI)
- [Tauri v2 prerequisites](https://v2.tauri.app/start/prerequisites/) (platform-specific system dependencies)

Install npm dependencies:

```bash
npm install
```

## Quick Start

This example bundles the **SenseVoice int8** model (model type 15), which
supports Chinese, English, Japanese, Korean, and Cantonese.

### 1. Download the model and Silero VAD

```bash
cd tauri-examples/non-streaming-speech-recognition-from-file

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

#### Optional: Homophone replacer

Some models (e.g. SenseVoice) support a homophone replacer for correcting
commonly confused words. To enable it, place `lexicon.txt` and `replace.fst`
directly inside `assets/`:

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
cp lexicon.txt replace.fst src-tauri/assets/
```

The app works fine without these files.

### 3. Run in development mode

```bash
npm run dev
```

This opens the app window. Click **Select Audio/Video File**, pick a file, and
recognition starts automatically.

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

You can also use the build script generator to automate this:

```bash
python scripts/tauri/generate-vad-asr.py --gen-registry
python scripts/tauri/generate-vad-asr.py --total 1 --index 0
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Frontend (HTML + JS)                            │
│  index.html / main.js / styles.css               │
│                                                  │
│  invoke("recognize_file", {path})                │
│  setInterval → invoke("get_recognition_progress")│
└──────────────┬───────────────────────────────────┘
               │ Tauri IPC (invoke)
┌──────────────▼───────────────────────────────────┐
│  Backend (Rust)                                  │
│                                                  │
│  lib.rs                                          │
│  ├── recognize_file()  → spawns worker thread    │
│  ├── run_recognition() → decode → VAD → ASR      │
│  ├── get_recognition_progress() → poll state     │
│  ├── cancel_recognition()                        │
│  ├── export_srt()                                │
│  └── save_segment_as_wav()                       │
│                                                  │
│  model_registry.rs (auto-generated, 62 models)   │
└──────────────────────────────────────────────────┘
```

**Processing pipeline:**

1. **symphonia** decodes audio/video packets to mono f32 PCM (streaming, not buffered).
2. **LinearResampler** resamples to 16 kHz if the native sample rate differs.
3. **Silero VAD** processes 512-sample (32 ms) windows and detects speech segments.
4. **OfflineRecognizer** transcribes each speech segment.
5. Results are accumulated and polled by the frontend every 200 ms.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Models are still loading" on startup | Models load in a background thread. Wait a few seconds. Large models take longer. |
| "Failed to create recognizer" | Check that `src-tauri/assets/<MODEL_NAME>/` exists and contains the expected `.onnx` and `tokens.txt` files. |
| "Unsupported format" | The audio format may not be enabled in `Cargo.toml`'s symphonia features. Check the `[dependencies.symphonia]` section. |
| Black rectangle in player | Normal for audio-only files — the `<video>` element shows controls but no picture. |
| App crashes on startup | Run `npm run dev` and check stderr for `[init]` or `[build_models]` log lines. |

## How It Works

1. Click **Select Audio/Video File** to choose a file via the native dialog.
2. The file is decoded packet-by-packet by symphonia to mono f32 PCM samples.
3. Audio is resampled to 16 kHz (required by Silero VAD).
4. Audio is fed to the VAD in 512-sample (32 ms) chunks.
5. Each detected speech segment (> 100 ms) is decoded by the offline recognizer.
6. Results are displayed in a table with start/end timestamps and text.
7. Click any row to seek the player to that segment. Subtitles overlay the player.
