# vad-from-file

This example demonstrates how to use Voice Activity Detection (VAD) from a file in Flutter with sherpa-onnx.

It uses the [Silero VAD v4](https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html) model.

It works on the following platforms:

  - Android
  - iOS
  - Linux
  - macOS (both arm64 and x86_64 are supported)
  - Windows
  - Web

## Features

- Configurable VAD parameters (threshold, min silence, min speech, max speech)
- Progress bar showing processing status
- Displays elapsed time, audio duration, and RTF after processing
- Click a segment to play it back

## How to build

### 1. Download the model

```bash
cd flutter-examples/vad-from-file/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/silero_vad.onnx
mkdir -p silero_vad
mv silero_vad.onnx silero_vad/
cd ..

./generate-asset-list.py
```

Note: `generate-asset-list.py` is a symlink to `../tts/generate-asset-list.py`.

### 2. Build the APP

  - For Linux

Install `libmpv-dev` first (required by the `media_kit` audio/video player):

```bash
sudo apt-get install -y libmpv-dev
flutter build linux
```

  - For macOS

```bash
flutter build macos
```

  - For Windows

```bash
flutter build windows
```

  - For Android

```bash
flutter build apk --split-per-abi --target-platform android-arm64
```

  - For web

```bash
flutter run -d chrome
```

  - For iOS

Connect your iPhone, then:

```bash
flutter devices
flutter run -d <device-id> --release
```
