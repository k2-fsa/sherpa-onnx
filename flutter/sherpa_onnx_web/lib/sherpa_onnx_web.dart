// Copyright (c)  2026  Xiaomi Corporation
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/services.dart';
import 'package:flutter_web_plugins/flutter_web_plugins.dart';

void _log(String message) {
  if (kDebugMode) {
    print('[sherpa_onnx_web] $message');
  }
}

/// Evaluate JavaScript source code in the global scope.
void _evalJs(String source) {
  final eval = globalContext.getProperty('eval'.toJS) as JSFunction;
  eval.callAsFunction(null, source.toJS);
}

/// Load a JS asset and return its source code.
Future<String> _loadAsset(String assetPath) async {
  final data = await rootBundle.load(assetPath);
  return utf8.decode(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
}

class SherpaOnnxWeb {
  static bool _initialized = false;
  static Completer<void>? _loadingCompleter;

  static void registerWith(Registrar registrar) {}

  /// Load the sherpa-onnx WASM module and JS wrappers.
  ///
  /// This must be called once before using any sherpa-onnx API on web.
  /// Typically called via `initBindingsAsync()` from the sherpa_onnx package.
  /// Safe to call concurrently — only the first call performs initialization.
  static Future<void> loadWasm() async {
    if (_initialized) return;

    // If another call is already in progress, wait for it.
    if (_loadingCompleter != null) {
      return _loadingCompleter!.future;
    }

    _loadingCompleter = Completer<void>();

    try {

    const prefix = 'packages/sherpa_onnx_web/assets';

    // 1. Load all JS assets (async I/O).
    _log('Loading JS assets...');
    final glueSource = await _loadAsset('$prefix/sherpa-onnx-wasm-web.js');
    final wrapperSources = await Future.wait([
      _loadAsset('$prefix/sherpa-onnx-asr.js'),
      _loadAsset('$prefix/sherpa-onnx-tts.js'),
      _loadAsset('$prefix/sherpa-onnx-vad.js'),
      _loadAsset('$prefix/sherpa-onnx-kws.js'),
      _loadAsset('$prefix/sherpa-onnx-punctuation.js'),
      _loadAsset('$prefix/sherpa-onnx-speaker-diarization.js'),
      _loadAsset('$prefix/sherpa-onnx-speech-enhancement.js'),
    ]);

    // 2. Load WASM binary.
    _log('Loading WASM binary...');
    final wasmData =
        await rootBundle.load('$prefix/sherpa-onnx-wasm-web.wasm');
    final wasmBytes = wasmData.buffer.asUint8List(wasmData.offsetInBytes, wasmData.lengthInBytes);

    // 3. Evaluate JS glue code (defines SherpaOnnx factory).
    _log('Evaluating JS...');
    _evalJs(glueSource);

    // 4. Define `module` for browser compatibility.
    //    The JS wrappers use `module.exports` (Node.js pattern).
    //    In the browser, `module` is not defined, so we stub it.
    _evalJs('if (typeof module === "undefined") { var module = {}; }');

    // 5. Evaluate JS wrappers (they use Module._FunctionName).
    for (final src in wrapperSources) {
      _evalJs(src);
    }

    // 6. Call the factory with wasmBinary.
    final factory =
        globalContext.getProperty('SherpaOnnx'.toJS) as JSFunction?;
    if (factory == null) {
      throw StateError('SherpaOnnx factory not found on globalThis');
    }

    final moduleConfig = JSObject();
    moduleConfig['wasmBinary'] = wasmBytes.toJS;
    var result = factory.callAsFunction(null, moduleConfig);

    // 7. Unwrap promises.
    while (result != null && result.isA<JSPromise>()) {
      result = await (result as JSPromise).toDart as JSObject?;
    }

    if (result == null || !result.isA<JSObject>()) {
      throw StateError('Failed to instantiate Emscripten Module');
    }

    // 8. Set Module as global so dart:js_interop code can access it.
    globalContext['Module'] = result;
    _log('WASM module initialized');

    _initialized = true;
    _loadingCompleter!.complete();
    } catch (e) {
      _loadingCompleter!.completeError(e);
      _loadingCompleter = null;
      rethrow;
    }
  }
}
