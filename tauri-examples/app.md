# Pre-built Tauri apps

This page lists the pre-built [Tauri](https://v2.tauri.app/) apps of
sherpa-onnx. All apps support **Linux, macOS, Windows, Android, and iOS**.

Source code: [tauri-examples](https://github.com/k2-fsa/sherpa-onnx/tree/master/tauri-examples)

## Table of contents

- [Tauri hello world](#tauri-hello-world)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Windows](#windows)
  - [Android](#android)
  - [iOS](#ios)
    - [Run iOS simulator example](#run-ios-simulator-example)
    - [Run iOS device example](#run-ios-device-example)
- [Doubao ASR from files](#doubao-asr-from-files)
  - [macOS](#macos-1)
  - [Linux](#linux-1)
  - [Windows](#windows-1)
  - [Android](#android-1)
  - [iOS](#ios-1)
    - [Run iOS simulator example (doubao-asr-file)](#run-ios-simulator-example-doubao-asr-file)
    - [Run iOS device example (doubao-asr-file)](#run-ios-device-example-doubao-asr-file)

## Tauri hello world

A minimal [Tauri](https://v2.tauri.app/) app that displays sherpa-onnx version
information.

Source code: [tauri-examples/hello_world](https://github.com/k2-fsa/sherpa-onnx/tree/master/tauri-examples/hello_world)

### macOS

| Platform | Link mode | File |
|---|---|---|
| macOS arm64 | static | [tauri-hello-world-macos-arm64-static.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-macos-arm64-static.zip) |
| macOS arm64 | shared | [tauri-hello-world-macos-arm64-shared.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-macos-arm64-shared.zip) |
| macOS x64 | static | [tauri-hello-world-macos-x64-static.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-macos-x64-static.zip) |
| macOS x64 | shared | [tauri-hello-world-macos-x64-shared.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-macos-x64-shared.zip) |

### Linux

| Platform | Link mode | File |
|---|---|---|
| Linux x64 | static | [tauri-hello-world-linux-x64-static.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-linux-x64-static.zip) |
| Linux x64 | shared | [tauri-hello-world-linux-x64-shared.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-linux-x64-shared.zip) |
| Linux arm64 | static | [tauri-hello-world-linux-arm64-static.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-linux-arm64-static.zip) |
| Linux arm64 | shared | [tauri-hello-world-linux-arm64-shared.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-linux-arm64-shared.zip) |

### Windows

| Platform | Link mode | File |
|---|---|---|
| Windows x64 | static | [tauri-hello-world-win-x64-static.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-win-x64-static.zip) |
| Windows x64 | shared | [tauri-hello-world-win-x64-shared.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-win-x64-shared.zip) |
| Windows arm64 | static | [tauri-hello-world-win-arm64-static.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-win-arm64-static.zip) |
| Windows arm64 | shared | [tauri-hello-world-win-arm64-shared.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-win-arm64-shared.zip) |

### Android

| Platform | File |
|---|---|
| Android arm64 | [tauri-hello-world-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-arm64-v8a.apk) |
| Android armv7 | [tauri-hello-world-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-armeabi-v7a.apk) |
| Android x86_64 | [tauri-hello-world-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-x86_64.apk) |

### iOS

| Platform | File |
|---|---|
| iOS device (arm64) | [tauri-hello-world-ios-device.ipa](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-ios-device.ipa) |
| iOS simulator (arm64) | [tauri-hello-world-ios-simulator.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-hello-world-ios-simulator.zip) |

#### Run iOS simulator example

```bash
# Download and unzip
unzip tauri-hello-world-ios-simulator.zip

# Boot a simulator
xcrun simctl boot "iPhone 16"

# Open Simulator app
open -a Simulator

# Install and launch
xcrun simctl install "iPhone 16" hello_world.app
xcrun simctl launch "iPhone 16" com.k2fsa.sherpa.onnx.hello.world
```

#### Run iOS device example

The IPA is unsigned. To install on a real device, re-sign it with your Apple
Developer certificate:

```bash
# Re-sign the IPA
codesign --force --sign "iPhone Developer: Your Name (TEAMID)" \
  --entitlements entitlements.plist \
  tauri-hello-world-ios-device.ipa

# Install via Xcode (Window → Devices and Simulators) or ios-deploy:
brew install ios-deploy
ios-deploy --bundle tauri-hello-world-ios-device.ipa
```

## Doubao ASR from files

Generate subtitles for audio or video files with the
[Doubao](https://www.doubao.com/) recording-file recognition API and local
[Sherpa-onnx VAD](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)
(nothing else runs in the cloud).

- Enter a Doubao API key, pick an **audio or video** file, press Start — no
  other settings. Not sure how to get a key? See the
  [help on getting a Doubao API key](https://www.doubao.com/thread/xXWsjZf94yLksVLtN).
- Click any transcript segment to **play** it; subtitles are shown over the
  player. A range of the file can be recognized instead of the whole file.
- View the transcript in **3 formats** — SRT, timestamped TXT, plain TXT —
  and export each of them. On Android the results can also be **shared** to
  other apps (e.g. WeChat).
- Results are saved automatically as `<name>.srt` and `<name>.txt`
  (timestamped): into the **Downloads** folder on Android, or next to the
  input file on desktop. If they already exist, the app **loads them** and
  offers to delete them, so recognition is never paid for twice.
- Supported inputs — audio: `wav mp3 flac ogg oga m4a aac opus weba wma aiff
  aif`; video: `mp4 webm mov mkv avi m4v ogv 3gp`.
- New Volcengine accounts get **20 hours of free recognition**. For more,
  [buy hours](https://console.volcengine.com/speech/new/purchase?chargeType=prepaid&productQueryId=10039&projectName=default) and
  select the package **录音文件识别** (recording-file recognition) — not
  流式语音识别 (streaming speech recognition).
- **Privacy**: speech is uploaded to Volcengine for recognition; it is not
  processed entirely on your device.

Source code: [tauri-examples/doubao-asr-file](https://github.com/k2-fsa/sherpa-onnx/tree/master/tauri-examples/doubao-asr-file)

### macOS

| Platform | File |
|---|---|
| macOS arm64 | [tauri-doubao-asr-file-macos-arm64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-macos-arm64.zip) |
| macOS x64 | [tauri-doubao-asr-file-macos-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-macos-x64.zip) |

### Linux

| Platform | File |
|---|---|
| Linux x64 | [tauri-doubao-asr-file-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-linux-x64.zip) |
| Linux arm64 | [tauri-doubao-asr-file-linux-arm64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-linux-arm64.zip) |

### Windows

| Platform | File |
|---|---|
| Windows x64 | [tauri-doubao-asr-file-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-win-x64.zip) |
| Windows arm64 | [tauri-doubao-asr-file-win-arm64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-win-arm64.zip) |

### Android

| Platform | File |
|---|---|
| Android arm64 | [tauri-doubao-asr-file-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-arm64-v8a.apk) |
| Android armv7 | [tauri-doubao-asr-file-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-armeabi-v7a.apk) |
| Android x86_64 | [tauri-doubao-asr-file-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-x86_64.apk) |

### iOS

| Platform | File |
|---|---|
| iOS device (arm64) | [tauri-doubao-asr-file-ios-device.ipa](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-ios-device.ipa) |
| iOS simulator (arm64) | [tauri-doubao-asr-file-ios-simulator.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/tauri/tauri-doubao-asr-file-ios-simulator.zip) |

#### Run iOS simulator example (doubao-asr-file)

```bash
# Download and unzip
unzip tauri-doubao-asr-file-ios-simulator.zip

# Boot a simulator
xcrun simctl boot "iPhone 16"

# Open Simulator app
open -a Simulator

# Install and launch
xcrun simctl install "iPhone 16" doubao-asr-file.app
xcrun simctl launch "iPhone 16" com.k2fsa.sherpa.onnx.doubao.asr.file
```

#### Run iOS device example (doubao-asr-file)

The IPA is unsigned. To install on a real device, re-sign it with your Apple
Developer certificate:

```bash
# Re-sign the IPA
codesign --force --sign "iPhone Developer: Your Name (TEAMID)" \
  --entitlements entitlements.plist \
  tauri-doubao-asr-file-ios-device.ipa

# Install via Xcode (Window → Devices and Simulators) or ios-deploy:
brew install ios-deploy
ios-deploy --bundle tauri-doubao-asr-file-ios-device.ipa
```
