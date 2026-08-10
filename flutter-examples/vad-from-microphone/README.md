# vad-from-microphone

This example demonstrates how to use Voice Activity Detection (VAD) from a microphone in Flutter with sherpa-onnx.

It supports the following VAD models:
  - [Silero VAD v4](https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html)
  - [Ten VAD](https://k2-fsa.github.io/sherpa/onnx/vad/ten-vad.html)

It works on the following platforms:

  - Android
  - iOS
  - Linux
  - macOS (both arm64 and x86_64 are supported)
  - Windows
  - Web

## Features

- Configurable VAD parameters (threshold, min silence, min speech, max speech)
- Real-time speech detection with circle indicator (black = silent, red = speech)
- Segment counter
- Support for both Silero VAD and Ten VAD models

## How to build

### 1. Download the model

For Silero VAD:

```bash
cd flutter-examples/vad-from-microphone/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/silero_vad.onnx
cd ..
```

For Ten VAD:

```bash
cd flutter-examples/vad-from-microphone/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vad-models/ten-vad.onnx
cd ..
```

Then generate the asset list:

```bash
./generate-asset-list.py
```

### 2. Select the model

Edit `lib/model_config.dart` and set `selectedModelIndex`:
- `0` = Silero VAD v4 (default)
- `1` = Ten VAD

### 3. Build the APP

  - For Linux

```bash
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

```bash
flutter devices
flutter run -d <device-id> --release
```

## Fix for Linux

```bash
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libunwind-dev
```
