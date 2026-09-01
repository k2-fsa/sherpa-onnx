# Dart API Examples

This directory contains examples for the [sherpa_onnx](https://pub.dev/packages/sherpa_onnx) Dart API.

All examples can also be used in Flutter apps, even though they are pure Dart
CLI programs here.

> **Start here:** Read the [`version/`](./version) example first. It
> demonstrates all initialization patterns (sync, async, and isolate) that
> every other example depends on.

> **For Flutter users:** Read [`hello_world`](../flutter-examples/hello_world)
> for initialization, then [`vad-from-microphone`](../flutter-examples/vad-from-microphone)
> or [`vad-from-file`](../flutter-examples/vad-from-file) for how to copy
> model files from assets to a writable directory and build a complete app.

## Initialization

All examples use the same simple initialization — no extra files needed:

```dart
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

// Sync (Flutter and Dart CLI):
sherpa_onnx.initBindings();

// Or async (Flutter, Dart CLI, and web):
await sherpa_onnx.initBindingsAsync();
```

No path argument is required. The library auto-resolves the native library
location for both Flutter apps and pure Dart CLI programs.

**Isolates:** If you use Dart isolates, you must call `initBindings()` or
`initBindingsAsync()` in every isolate that uses sherpa-onnx. See
[`version/`](./version/) for examples.

## Using in Flutter apps

These examples use model files from disk directly (e.g., `./model.onnx`).
In Flutter apps, model files must be bundled as assets and copied to a
writable location before use, because Flutter apps run in a sandbox and
cannot access arbitrary file paths.

**Steps to use a model in Flutter:**

1. Add the model file to your `pubspec.yaml`:

   ```yaml
   flutter:
     assets:
       - assets/model.onnx
   ```

2. Copy the model from the asset bundle to a writable directory at runtime:

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

3. Pass the copied file path to the sherpa-onnx config:

   ```dart
   final modelPath = await copyAsset('assets/model.onnx', 'model.onnx');
   final config = sherpa_onnx.OfflineRecognizerConfig(
     model: sherpa_onnx.OfflineModelConfig(
       paraformer: sherpa_onnx.OfflineParaformerModelConfig(model: modelPath),
     ),
   );
   ```

For concrete Flutter examples, see
[`flutter-examples/`](../flutter-examples/) in the repository.

## Examples

| Directory | Description |
|-----------|-------------|
| [version](./version) | Version info — demonstrates sync, async, and isolate initialization |
| [vad](./vad) | Voice activity detection |
| [vad-with-non-streaming-asr](./vad-with-non-streaming-asr) | VAD with non-streaming speech recognition (useful for subtitles) |
| [non-streaming-asr](./non-streaming-asr) | Non-streaming (offline) speech recognition |
| [streaming-asr](./streaming-asr) | Streaming (online) speech recognition |
| [tts](./tts) | Text to speech |
| [speaker-diarization](./speaker-diarization) | Speaker diarization |
| [speaker-identification](./speaker-identification) | Speaker identification and verification |
| [spoken-language-identification](./spoken-language-identification) | Spoken language identification |
| [audio-tagging](./audio-tagging) | Audio tagging |
| [keyword-spotter](./keyword-spotter) | Keyword spotting |
| [add-punctuations](./add-punctuations) | Adding punctuations to text |
| [speech-enhancement-gtcrn](./speech-enhancement-gtcrn) | Speech enhancement/denoising with GTCRN |
| [speech-enhancement-dpdfnet](./speech-enhancement-dpdfnet) | Speech enhancement/denoising with DPDFNet (16 kHz family) |
| [streaming-speech-enhancement-gtcrn](./streaming-speech-enhancement-gtcrn) | Streaming speech enhancement with GTCRN |
| [streaming-speech-enhancement-dpdfnet](./streaming-speech-enhancement-dpdfnet) | Streaming speech enhancement with DPDFNet |

## Running an example

```bash
cd vad
dart pub get
dart run ./bin/vad.dart --help
```

## Creating a new example

```bash
dart create my-example
cd my-example

# Add sherpa_onnx to pubspec.yaml:
#   dependencies:
#     sherpa_onnx: ^1.13.7
#     path: ^1.9.0

dart pub get
```

In your `main.dart`:

```dart
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() async {
  await sherpa_onnx.initBindingsAsync();
  // Use sherpa-onnx APIs here...
}
```
