# Doubao ASR from audio/video files (Tauri)

This example generates subtitles for an audio or **video** file:

```
input media file
  -> decode the audio track (symphonia; ffmpeg fallback)
  -> sherpa-onnx silero-vad cuts speech segments   (local, offline)
  -> each segment is sent to the Doubao bigmodel recording-file API
  -> results are shown in the GUI and exported as subtitles
```

VAD runs locally. Only ASR is remote, via the Doubao (豆包) API.

## Features

- **No settings to configure.** Enter your API key, pick a file, press Start.
- **Recent files** (last 50, kept across restarts): pick one from the
  dropdown to select it again; a **Clear history** button wipes the list.
- Optionally limit recognition to a **range** of the file, e.g. only the first
  20 seconds (`from 0 s to 20 s`). By default the **whole file** is sent.
- Click any transcript segment to **play** it in the media preview.
- **Subtitles over the player** while playing (video and audio) — with a
  show/hide button, shown by default.
- **Stop** at any time — everything recognized so far is kept and can still be
  exported.
- Results are **auto-saved** as `<name>.srt` and `<name>.txt` (timestamped),
  like the Python example (`generate-subtitles-doubao.py`): into the public
  **Downloads** folder on Android, or next to the input file on desktop.
  Recognition is not free, so results are saved even when you stop early. If
  those files **already exist**, the app refuses to start (before any
  decoding, VAD or API call) and asks you to delete them first — and it
  **loads the saved transcript and shows it** right away, so previous results
  are never lost or paid for twice.
- View the transcript in **3 formats** (switchable tabs): **SRT**
  (`00:00:17,030 --> 00:00:19,942`), **timestamped TXT**
  (`0:00:17,030 --> 0:00:19,942  text`), **plain TXT** (just the text).
- **Search** the transcript: matching words are highlighted, the current hit is
  emphasized, and Enter / Shift+Enter (or the ↑/↓ buttons) jump between matches.
- Export the current results in all 3 formats.
- `silero_vad.onnx` is **bundled inside the app**; nothing to download manually.

## Supported input formats

Audio: `wav mp3 flac ogg oga m4a aac opus weba wma aiff aif`
Video: `mp4 webm mov mkv avi m4v ogv 3gp`

The audio track is extracted from video files. Most formats decode out of the
box (pure-Rust symphonia). For `opus`/`wma`/`avi` the app falls back to
`ffmpeg` if it is installed; otherwise it tells you the format is unsupported.

## Privacy

Speech recognition runs on Volcengine's servers, not on your device: the app
uploads each speech segment to the Doubao API and receives the transcript
back. Only VAD (speech segmentation) runs locally. Please take this into
account before processing sensitive or confidential audio.

## API key

The app asks for a Doubao API key on the main page and stores it only on your
computer. If you don't have one, please see the help:

**[How to get a Doubao API key](https://www.doubao.com/thread/xXWsjZf94yLksVLtN)**

**Free quota & buying more hours**

- When you first register on Volcengine, you get **20 hours of free
  recognition**.
- If you need more hours, buy them at
  <https://console.volcengine.com/speech/new/purchase?chargeType=prepaid&productQueryId=10039&projectName=default>
  and select the package **录音文件识别** (recording-file recognition).
  **Do not select 流式语音识别** (streaming speech recognition) — that is
  a different, wrong package for this app.

## Platforms

Tauri 2 builds for macOS, Windows, Linux, Android and iOS. Pre-built apps are
published on the
[tauri release tag](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tauri)
as `tauri-doubao-asr-file-*`.

## Build from source

Requirements: Rust (stable), Node.js, CMake, a C++ compiler.

```bash
# 1. Put the VAD model into the bundled assets (end users never do this —
#    the released app already contains it)
mkdir -p tauri-examples/doubao-asr-file/src-tauri/assets
curl -sSL -o tauri-examples/doubao-asr-file/src-tauri/assets/silero_vad.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

# 2. Build the app. sherpa-onnx is used as a normal Rust crate (static link by
#    default): sherpa-onnx-sys downloads the prebuilt native libraries for your
#    target, so there is no need to build sherpa-onnx from source.
cd tauri-examples/doubao-asr-file
npm install
npm run build
```

For Android, after `npx tauri android init` copy the helper plugin into the
generated project before building (it resolves real file names like
`1_lei-jun.wav` from picker URIs and saves results to the public Downloads
folder). The CI workflow does the same copy before Gradle runs:

```bash
d=src-tauri/gen/android/app/src/main/java/com/plugin/apphelper
mkdir -p "$d" && cp src-tauri/android/AppHelperPlugin.kt "$d/"
```

To use the in-repo Rust crates instead of crates.io:

```bash
mkdir -p src-tauri/.cargo
cat > src-tauri/.cargo/config.toml <<'EOF'
[patch.crates-io]
sherpa-onnx = { path = "../../../sherpa-onnx/rust/sherpa-onnx" }
sherpa-onnx-sys = { path = "../../../sherpa-onnx/rust/sherpa-onnx-sys" }
EOF
```

## Tests

```bash
cd src-tauri
cargo test
```

The tests cover subtitle formatting (all 3 formats) and the Doubao request
builders/response parsing — no network access or API key is needed.
