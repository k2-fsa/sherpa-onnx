# Non-Streaming Speech Recognition from File

A Tauri v2 desktop app that transcribes audio and video files using offline ASR
with Silero VAD.

## Features

- **62 ASR models** supported (SenseVoice, Paraformer, Whisper, Transducer, Moonshine, etc.)
- **Audio/video playback** with waveform display
- **SRT subtitle export**
- **Segment WAV export** — save individual speech segments as WAV files
- **Progress tracking** with cancellation support
- **Cross-platform** — macOS (universal), Linux (x64/aarch64), Windows (x64)
- **Pure-Rust audio decoding** via [symphonia](https://github.com/pdeljanov/Symphonia) — no system dependencies

## Supported Audio Formats

Any format supported by symphonia:

| Type   | Formats                                    |
|--------|--------------------------------------------|
| Audio  | MP3, FLAC, AAC, OGG/Vorbis, WAV, AIFF, ADPCM |
| Video  | MP4/M4A, MKV, WebM (audio track extracted) |

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable)
- [Tauri CLI](https://v2.tauri.app/start/prerequisites/):

```bash
cargo install tauri-cli
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

### 2. Copy resources into src-tauri

Tauri bundles files from `src-tauri/resources/`. Place the entire model
directory (keeping its original name) and `silero_vad.onnx` inside it:

```bash
mkdir -p src-tauri/resources
cp -a sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17 src-tauri/resources/
cp -a silero_vad.onnx src-tauri/resources/
```

This gives:

```
src-tauri/resources/
├── silero_vad.onnx
└── sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/
    ├── model.int8.onnx
    └── tokens.txt
```

Some models (e.g. SenseVoice) support a homophone replacer. To enable it,
place `lexicon.txt` and `rule.fst` directly inside `resources/`:

```bash
# Optional — only if you want homophone replacement
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
cp -a lexicon.txt replace.fst src-tauri/resources/
```

The app works fine without these files.

### 3. Build and run

```bash
cargo tauri dev
```

This opens the app window. Use the file picker to select an audio or video file
and click **Recognize**.

### 4. Build a release binary

```bash
cargo tauri build
```

The output is in `src-tauri/target/release/bundle/`.

## Using a Different Model

The app uses `MODEL_TYPE` and `MODEL_NAME` (constants in `src-tauri/src/lib.rs`)
to select which model to bundle. The defaults are `15` and
`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`.

To switch models:

1. Choose a model type from `src-tauri/src/model_registry.rs` (types 0–61).
2. Set `MODEL_TYPE` and `MODEL_NAME` in `src-tauri/src/lib.rs`:

```rust
const MODEL_TYPE: u32 = 42;
const MODEL_NAME: &str = "your-model-dir-name";
```

3. Download the corresponding model and place the entire directory into
   `src-tauri/resources/`, keeping its original name.

You can also use the build script generator to automate this:

```bash
# Generate the model registry from model definitions
python scripts/tauri/generate-vad-asr.py --gen-registry

# Generate a build script for a specific model shard
python scripts/tauri/generate-vad-asr.py --total 1 --index 0
```

## How It Works

1. Click **Select Audio File** to choose an audio/video file via native dialog.
2. The file is decoded by symphonia to mono f32 PCM samples.
3. Audio is resampled to 16 kHz (required by Silero VAD).
4. Audio is fed to the VAD in 512-sample (32 ms) chunks.
5. Each detected speech segment is decoded by the offline recognizer.
6. Results are displayed in a table with start/end timestamps and text.
