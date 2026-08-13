// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io' show Platform;
import 'dart:isolate' show Isolate;

// Conditional import: native uses dart:ffi, web uses dart:js_interop.
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
/// ## Initialization
///
/// Before creating any runtime object, call [initBindings] (sync, native only)
/// or [initBindingsAsync] (async, all platforms including web) once to load
/// the underlying native `sherpa-onnx-c-api` library. No path argument is
/// needed — the library auto-resolves the native library location for both
/// Flutter apps and pure Dart CLI programs.
///
/// ```dart
/// import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
///
/// // Sync (Flutter and Dart CLI, not web):
/// sherpa_onnx.initBindings();
///
/// // Async (Flutter, Dart CLI, and web):
/// await sherpa_onnx.initBindingsAsync();
/// ```
///
/// ## Isolates
///
/// Each isolate has its own FFI binding state. You **must** call
/// [initBindings] or [initBindingsAsync] in every isolate that uses
/// sherpa-onnx APIs. Calling it in one isolate does NOT make sherpa-onnx
/// available in other isolates.
///
/// ## Examples
///
/// For concrete end-to-end usage, see `dart-api-examples/` in the repository,
/// especially:
///
/// - `version/` — version info with sync, async, and isolate examples
/// - `vad/` — voice activity detection
/// - `non-streaming-asr/` — offline speech recognition
/// - `streaming-asr/` — streaming speech recognition
/// - `tts/` — text-to-speech
/// - `speaker-diarization/` — speaker diarization

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

// On native platforms (dart:io available), this is always false.
// On web (dart:js_interop available), the web init.dart is loaded instead.
const bool _kIsWeb = bool.fromEnvironment('dart.library.io') == false;

String? _resolvePlatform() {
  if (Platform.isMacOS) return 'macos';
  if (Platform.isLinux) return 'linux';
  if (Platform.isWindows) return 'windows';
  return null;
}

String? _resolvePath(String platform, Uri uri) {
  final filePath = uri.toFilePath();
  final sep = Platform.pathSeparator;
  final libDir = filePath.substring(0, filePath.lastIndexOf(sep));
  final parentDir = libDir.endsWith('${sep}lib')
      ? libDir.substring(0, libDir.length - 4)
      : libDir;

  if (Platform.isLinux) {
    final arch = Platform.version.contains('arm64') ||
            Platform.version.contains('aarch64')
        ? 'aarch64'
        : 'x64';
    return '$parentDir${sep}$platform$sep$arch';
  }

  return '$parentDir$sep$platform';
}

/// Resolve the path to the sherpa-onnx native library (async).
///
/// Uses [Isolate.resolvePackageUri] to locate the platform-specific
/// sherpa_onnx package directory (e.g., `sherpa_onnx_macos`).
///
/// Throws [UnsupportedError] on unsupported platforms and [StateError] if
/// the platform package cannot be found. Not supported in Flutter — use
/// [initBindings] or [initBindingsAsync] instead (they handle this
/// automatically).
Future<String> resolveSherpaOnnxPath() async {
  final platform = _resolvePlatform();
  if (platform == null) {
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  final uri = await Isolate.resolvePackageUri(
      Uri.parse('package:sherpa_onnx_$platform/any_path_is_ok_here.dart'));

  if (uri == null) {
    throw StateError(
      'Could not resolve package:sherpa_onnx_$platform. '
      'Make sure sherpa_onnx_$platform is a dependency.',
    );
  }

  return _resolvePath(platform, uri)!;
}

/// Resolve the path to the sherpa-onnx native library (sync).
///
/// Uses [Isolate.resolvePackageUriSync] to locate the platform-specific
/// sherpa_onnx package directory. Returns `null` if not supported (e.g.,
/// Flutter) or if the package cannot be found.
String? resolveSherpaOnnxPathSync() {
  try {
    final platform = _resolvePlatform();
    if (platform == null) return null;

    final uri = Isolate.resolvePackageUriSync(
        Uri.parse('package:sherpa_onnx_$platform/any_path_is_ok_here.dart'));

    if (uri == null) return null;
    return _resolvePath(platform, uri);
  } catch (_) {
    // Not supported in Flutter.
    return null;
  }
}

/// Initialize the native sherpa-onnx bindings synchronously.
///
/// Works on Flutter and Dart CLI without a path argument — the native
/// library location is auto-resolved. On Flutter, uses
/// `DynamicLibrary.process()`; on Dart CLI, locates the library in the
/// pub cache via [resolveSherpaOnnxPathSync].
///
/// Does **not** support web — use [initBindingsAsync] instead.
///
/// **Isolates:** Each isolate has its own FFI binding state. You must call
/// [initBindings] or [initBindingsAsync] in every isolate that uses
/// sherpa-onnx APIs.
void initBindings([String? p]) {
  if (_kIsWeb) {
    throw UnsupportedError(
      'initBindings() is not supported on web. '
      'Use initBindingsAsync() instead.',
    );
  }
  p ??= resolveSherpaOnnxPathSync();
  _path ??= p;
  init.initNativeBindings(_path);
}

/// Initialize the sherpa-onnx bindings asynchronously.
///
/// Works on all platforms including web. On native platforms, the native
/// library path is auto-resolved (no argument needed). On web, loads the
/// WASM module from bundled assets.
///
/// **Isolates:** Each isolate has its own FFI binding state. You must call
/// [initBindings] or [initBindingsAsync] in every isolate that uses
/// sherpa-onnx APIs.
Future<void> initBindingsAsync([String? p]) async {
  if (p == null && !_kIsWeb) {
    try {
      p = await resolveSherpaOnnxPath();
    } catch (_) {
      // Isolate.resolvePackageUri is not supported in Flutter.
      // Fall back to DynamicLibrary.process() via initNativeBindings(null).
    }
  }
  _path ??= p;
  if (_kIsWeb) {
    await web.SherpaOnnxWeb.loadWasm();
    return;
  }
  init.initNativeBindings(_path);
}
