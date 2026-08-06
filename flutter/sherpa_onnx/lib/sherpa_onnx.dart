// Copyright (c)  2024  Xiaomi Corporation
import 'package:flutter/foundation.dart' show kIsWeb;

// Conditional import: native uses dart:io/dart:ffi, web uses dart:js_interop.
import 'src/init_native.dart'
    if (dart.library.js_interop) 'src/web/init.dart' as init;

// Conditional import for web WASM loader.
import 'package:sherpa_onnx_web/sherpa_onnx_web.dart'
    if (dart.library.io) 'src/init_stub.dart' as web;

/// Dart bindings for the public sherpa-onnx inference APIs.
///
/// Import this library to access offline and streaming ASR, text-to-speech,
/// VAD, speaker identification, speaker diarization, punctuation restoration,
/// audio tagging, spoken language identification, speech denoising, and WAV
/// I/O helpers from a single entry point.
///
/// Before creating any runtime object, call [initBindings] (native) or
/// [initBindingsAsync] (all platforms including web) once so the package can
/// load the underlying native `sherpa-onnx-c-api` library for the current
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

export 'src/audio_tagging_config.dart';
export 'src/audio_tagging.dart'
    if (dart.library.js_interop) 'src/web/audio_tagging.dart';
export 'src/feature_config.dart';
export 'src/homophone_replacer_config.dart';
export 'src/keyword_spotter_config.dart';
export 'src/keyword_spotter.dart'
    if (dart.library.js_interop) 'src/web/keyword_spotter.dart';
export 'src/offline_punctuation_config.dart';
export 'src/offline_punctuation.dart'
    if (dart.library.js_interop) 'src/web/offline_punctuation.dart';
export 'src/offline_recognizer_config.dart';
export 'src/offline_recognizer.dart'
    if (dart.library.js_interop) 'src/web/offline_recognizer.dart';
export 'src/offline_speaker_diarization_config.dart';
export 'src/offline_speaker_diarization.dart'
    if (dart.library.js_interop) 'src/web/offline_speaker_diarization.dart';
export 'src/offline_speech_denoiser_config.dart';
export 'src/offline_speech_denoiser.dart'
    if (dart.library.js_interop) 'src/web/offline_speech_denoiser.dart';
export 'src/offline_stream.dart'
    if (dart.library.js_interop) 'src/web/offline_stream.dart';
export 'src/online_speech_denoiser_config.dart';
export 'src/online_speech_denoiser.dart'
    if (dart.library.js_interop) 'src/web/online_speech_denoiser.dart';
export 'src/online_punctuation_config.dart';
export 'src/online_punctuation.dart'
    if (dart.library.js_interop) 'src/web/online_punctuation.dart';
export 'src/online_recognizer_config.dart';
export 'src/online_recognizer.dart'
    if (dart.library.js_interop) 'src/web/online_recognizer.dart';
export 'src/online_stream.dart'
    if (dart.library.js_interop) 'src/web/online_stream.dart';
export 'src/speaker_identification_config.dart';
export 'src/speaker_identification.dart'
    if (dart.library.js_interop) 'src/web/speaker_identification.dart';
export 'src/spoken_language_identification_config.dart';
export 'src/spoken_language_identification.dart'
    if (dart.library.js_interop) 'src/web/spoken_language_identification.dart';
export 'src/tts_config.dart';
export 'src/tts.dart'
    if (dart.library.js_interop) 'src/web/tts.dart';
export 'src/vad_config.dart';
export 'src/vad.dart'
    if (dart.library.js_interop) 'src/web/vad.dart';
export 'src/version.dart'
    if (dart.library.js_interop) 'src/web/version.dart';
export 'src/wave_reader_config.dart';
export 'src/wave_reader.dart'
    if (dart.library.js_interop) 'src/web/wave_reader.dart';
export 'src/wave_writer.dart'
    if (dart.library.js_interop) 'src/web/wave_writer.dart';

String? _path;

/// Initialize the native sherpa-onnx bindings.
///
/// **Important:** This must be called in every isolate that uses sherpa-onnx.
/// Each isolate has its own FFI binding state, so calling `initBindings()` in
/// one isolate does NOT make sherpa-onnx available in other isolates. If you
/// use Dart isolates for background work (e.g., TTS generation, model loading),
/// call `initBindings()` in each isolate before calling any sherpa-onnx API.
///
/// On web, use [initBindingsAsync] instead. This method throws
/// [UnsupportedError] on web.
void initBindings([String? p]) {
  if (kIsWeb) {
    throw UnsupportedError(
      'initBindings() is not supported on web. '
      'Use initBindingsAsync() instead.',
    );
  }
  _path ??= p;
  init.initNativeBindings(_path);
}

/// Initialize the sherpa-onnx bindings (works on all platforms including web).
///
/// On web, this loads the WASM module and JS wrappers automatically.
/// On native platforms, this behaves the same as [initBindings].
///
/// **Important:** If you use Dart isolates, call `initBindings()` or
/// `initBindingsAsync()` in each isolate that calls sherpa-onnx APIs.
/// See [initBindings] for details.
Future<void> initBindingsAsync([String? p]) async {
  _path ??= p;
  if (kIsWeb) {
    await web.SherpaOnnxWeb.loadWasm();
    return;
  }
  init.initNativeBindings(_path);
}
