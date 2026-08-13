# Flutter Examples

This directory contains Flutter examples for [sherpa-onnx](https://pub.dev/packages/sherpa_onnx).

> **Start here:** Read [`hello_world`](./hello_world) first to learn how to
> initialize sherpa-onnx in a Flutter app. Then read
> [`vad-from-microphone`](./vad-from-microphone) or
> [`vad-from-file`](./vad-from-file) to learn how to copy model files from
> assets to a writable directory — this is required for all examples that use
> models.

## Examples

| Directory | Description |
|-----------|-------------|
| [hello_world](./hello_world) | Minimal example — prints sherpa-onnx version info |
| [streaming_asr](./streaming_asr) | Streaming (online) speech recognition |
| [non_streaming_vad_asr](./non_streaming_vad_asr) | VAD + non-streaming speech recognition |
| [tts](./tts) | Text to speech |
| [vad-from-file](./vad-from-file) | Voice activity detection from an audio file |
| [vad-from-microphone](./vad-from-microphone) | Voice activity detection from microphone |
| [offline-punctuation](./offline-punctuation) | Offline punctuation restoration |
| [online-punctuation](./online-punctuation) | Online punctuation restoration |

## Initialization

All Flutter examples initialize sherpa-onnx the same way:

```dart
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

await sherpa_onnx.initBindingsAsync();
```

No path argument is needed. The native library is linked into the app bundle
by the Flutter build system.

**Isolates:** Each isolate has its own FFI binding state. You **must** call
`initBindings()` or `initBindingsAsync()` in every isolate that uses
sherpa-onnx APIs. Calling it in one isolate does NOT make sherpa-onnx
available in other isolates. See the
[`vad-from-microphone`](./vad-from-microphone) example for a working
isolate pattern.

## Model files

Flutter apps run in a sandbox and cannot access arbitrary file paths. Model
files must be bundled as Flutter assets and copied to a writable directory
at runtime.

**Step 1:** Add the model to `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/model.onnx
```

**Step 2:** Copy the model to a writable directory:

```dart
import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

Future<String> copyAsset(String assetPath, String fileName) async {
  final dir = await getApplicationDocumentsDirectory();
  final file = File('${dir.path}/$fileName');
  if (!await file.exists()) {
    final data = await rootBundle.load(assetPath);
    await file.writeAsBytes(data.buffer.asUint8List());
  }
  return file.path;
}
```

**Step 3:** Use the copied path in your config:

```dart
final modelPath = await copyAsset('assets/model.onnx', 'model.onnx');
```

See [`vad-from-microphone`](./vad-from-microphone) or
[`vad-from-file`](./vad-from-file) for complete working examples.

## Running on different platforms

### macOS

```bash
cd hello_world
flutter pub get
flutter run -d macos
```

### iOS

Connect your iPhone, then:

```bash
flutter run -d <device-id>
```

You need a valid Apple Developer certificate for device deployment.
For iOS apps that use the microphone, add `NSMicrophoneUsageDescription`
to `ios/Runner/Info.plist`.

### Android

```bash
flutter run -d <device-id>
```

If you get a `minSdk` error, update `android/app/build.gradle`:

```gradle
android {
    defaultConfig {
        minSdk = 23
    }
}
```

### Linux

```bash
flutter config --enable-linux-desktop
flutter run -d linux
```

### Windows

```bash
flutter run -d windows
```

### Web

```bash
flutter run -d chrome
```

## Pre-built apps

Download pre-built Flutter apps for different platforms at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/flutter

## Web demos

Try the following demos directly in your browser:

| Demo | URL |
|---|---|
| VAD from file | https://modelscope.cn/studios/csukuangfj/wasm-vad-from-file |
| VAD from microphone | https://modelscope.cn/studios/csukuangfj/wasm-vad-from-microphone |
| Offline punctuation | https://modelscope.cn/studios/csukuangfj/wasm-offline-punctuation |
| Online punctuation | https://modelscope.cn/studios/csukuangfj/wasm-online-punctuation |
