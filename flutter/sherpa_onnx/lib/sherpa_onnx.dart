// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';
import 'dart:ffi';

/// Dart bindings for the public sherpa-onnx inference APIs.
///
/// Import this library to access offline and streaming ASR, text-to-speech,
/// VAD, speaker identification, speaker diarization, punctuation restoration,
/// audio tagging, spoken language identification, speech denoising, and WAV
/// I/O helpers from a single entry point.
///
/// Before creating any runtime object, call [initBindings] once so the package
/// can load the underlying native `sherpa-onnx-c-api` library for the current
/// platform.
///
/// For concrete end-to-end usage, see `dart-api-examples/` in the repository,
/// especially:
///
/// - `non-streaming-asr/bin/sense-voice.dart`
/// - `non-streaming-asr/bin/whisper.dart`
/// - `non-streaming-asr/bin/nemo-transducer.dart`
/// - `streaming-asr/bin/zipformer-transducer.dart`
/// - `tts/bin/pocket-en.dart`
/// - `vad/bin/vad.dart`
/// - `speaker-diarization/`

export 'src/audio_tagging.dart';
export 'src/feature_config.dart';
export 'src/homophone_replacer_config.dart';
export 'src/keyword_spotter.dart';
export 'src/offline_punctuation.dart';
export 'src/offline_recognizer.dart';
export 'src/offline_speaker_diarization.dart';
export 'src/offline_speech_denoiser.dart';
export 'src/offline_stream.dart';
export 'src/online_speech_denoiser.dart';
export 'src/online_punctuation.dart';
export 'src/online_recognizer.dart';
export 'src/online_stream.dart';
export 'src/speaker_identification.dart';
export 'src/spoken_language_identification.dart';
export 'src/tts.dart';
export 'src/vad.dart';
export 'src/version.dart';
export 'src/wave_reader.dart';
export 'src/wave_writer.dart';

import 'src/sherpa_onnx_bindings.dart';

String? _path;

// see also
// https://github.com/flutter/codelabs/blob/main/ffigen_codelab/step_05/lib/ffigen_app.dart
// https://api.flutter.dev/flutter/dart-io/Platform-class.html
final DynamicLibrary _dylib = () {
  if (Platform.isMacOS) {
    if (_path == null) {
      return DynamicLibrary.open('libsherpa-onnx-c-api.dylib');
    } else {
      return DynamicLibrary.open('$_path/libsherpa-onnx-c-api.dylib');
    }
  }

  if (Platform.isIOS) {
    if (_path == null) {
      return DynamicLibrary.open('sherpa_onnx.framework/sherpa_onnx');
    } else {
      return DynamicLibrary.open('$_path/sherpa_onnx.framework/sherpa_onnx');
    }
  }

  if (Platform.isAndroid || Platform.isLinux) {
    if (_path == null) {
      return DynamicLibrary.open('libsherpa-onnx-c-api.so');
    } else {
      return DynamicLibrary.open('$_path/libsherpa-onnx-c-api.so');
    }
  }

  if (Platform.isWindows) {
    if (_path == null) {
      return DynamicLibrary.open('sherpa-onnx-c-api.dll');
    } else {
      return DynamicLibrary.open('$_path\\sherpa-onnx-c-api.dll');
    }
  }

  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

/// Initialize the native sherpa-onnx bindings.
///
/// Call this exactly once before using any other API from this package.
///
/// If [p] is provided, it is treated as the directory containing the native
/// dynamic library for desktop platforms, or the framework root on Apple
/// platforms. If omitted, the package tries to load the library from the
/// default platform-specific filename.
void initBindings([String? p]) {
  _path ??= p;
  SherpaOnnxBindings.init(_dylib);
}
