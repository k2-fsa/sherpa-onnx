# Non-Streaming Speech Recognition from File (Tauri)

A Tauri desktop application that performs speech recognition on audio and video
files using [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

It uses a VAD (Voice Activity Detector) to split long audio into speech
segments, then recognizes each segment with an offline ASR model.

## Supported Formats

Audio decoding uses [symphonia](https://github.com/pdeljanov/Symphonia) — a
pure Rust decoder with **no system dependencies** (no ffmpeg needed). Supported
formats include:

| Type   | Formats                                    |
|--------|--------------------------------------------|
| Audio  | MP3, FLAC, AAC, OGG/Vorbis, WAV, AIFF, ADPCM |
| Video  | MP4/M4A, MKV, WebM (audio track extracted) |

## Prerequisites

1. Install [Rust](https://rustup.rs/) and [Node.js](https://nodejs.org/).
2. Install Tauri prerequisites: https://tauri.app/start/prerequisites/
3. Download a SenseVoice model and Silero VAD model:

```bash
cd src-tauri

# SenseVoice model (multilingual, zh/en/ja/ko)
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

# Silero VAD model
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

4. Update model paths in `src-tauri/src/lib.rs` if you placed models in a
   different location.

5. Generate app icons (optional, for bundling):

```bash
npm install
npm run tauri icon /path/to/icon.png
```

## Run in Development Mode

```bash
cd src-tauri
cargo tauri dev
```

Or from the project root:

```bash
npm install
npm run dev
```

## Build for Production

```bash
cd src-tauri
cargo tauri build
```

## How It Works

1. Click **Select Audio File** to choose an audio/video file via native dialog.
2. The file is decoded by symphonia to mono f32 PCM samples.
3. Audio is resampled to 16 kHz (required by Silero VAD).
4. Audio is fed to the VAD in 512-sample (32 ms) chunks.
5. Each detected speech segment is decoded by the offline recognizer.
6. Results are displayed in a table with start/end timestamps and text.

The VAD + ASR logic mirrors the C++ example in
`sherpa-onnx/csrc/sherpa-onnx-vad-with-offline-asr.cc`.

## Using a Different Model

Edit the model configuration section in `src-tauri/src/lib.rs`. For example,
to use a Whisper model instead:

```rust
asr_config.model_config.whisper = OfflineWhisperModelConfig {
    encoder: Some("./whisper-encoder.onnx".into()),
    decoder: Some("./whisper-decoder.onnx".into()),
    language: Some("en".into()),
    task: Some("transcribe".into()),
    tail_paddings: 200,
    ..Default::default()
};
```

See the [sherpa-onnx documentation](https://k2-fsa.github.io/sherpa/onnx/)
for all supported model types.
