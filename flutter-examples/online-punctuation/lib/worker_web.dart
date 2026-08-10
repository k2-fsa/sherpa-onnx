// Web Worker support for punctuation.
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;

import './model_web.dart' as m;

typedef OnReadyCallback = void Function();
typedef OnResultCallback = void Function(String result, double elapsed);
typedef OnErrorCallback = void Function(String message);

/// Manages a Web Worker for punctuation.
class PunctWorker {
  web.Worker? _worker;
  final OnReadyCallback onReady;
  final OnResultCallback onResult;
  final OnErrorCallback onError;

  PunctWorker({
    required this.onReady,
    required this.onResult,
    required this.onError,
  });

  /// Initialize the worker: load WASM and model files, send to worker.
  Future<void> init() async {
    final modelFiles = await m.loadModelFileBytes();
    final config = await m.prepareModelConfig();

    if (kDebugMode) {
      print('[worker_web] config: ${config.toString()}');
      print('[worker_web] modelFiles: ${modelFiles.length} files');
    }

    // Create Web Worker.
    _worker = web.Worker('punct-worker.js'.toJS);

    // Listen for messages from the worker.
    _worker!.onmessage = (web.MessageEvent event) {
      _handleMessage(event);
    }.toJS;

    // Handle worker startup failures.
    _worker!.onerror = (web.ErrorEvent event) {
      onError('Worker error: ${event.message}');
    }.toJS;

    // Load JS glue source, punctuation helpers, and WASM binary.
    final jsGlueSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.js');
    final punctJsSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-punctuation.js');
    final wasmData = await _loadAssetBytes(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.wasm');

    // Build model files map.
    final jsModelFiles = JSObject();
    for (final entry in modelFiles.entries) {
      final bytes = Uint8List.fromList(entry.value);
      jsModelFiles[entry.key] = bytes.buffer.toJS;
    }

    // Convert config to JS format.
    final jsConfig = m.configToJs(config);

    // Send init message.
    final initMsg = JSObject();
    initMsg['type'] = 'init'.toJS;
    initMsg['jsGlueSource'] = jsGlueSource.toJS;
    initMsg['punctJsSource'] = punctJsSource.toJS;
    initMsg['wasmBinary'] = wasmData.buffer.toJS;
    initMsg['modelFiles'] = jsModelFiles;
    initMsg['config'] = jsConfig;
    _worker!.postMessage(initMsg);
  }

  /// Punctuate text.
  void punctuate({required String text}) {
    final msg = JSObject();
    msg['type'] = 'punctuate'.toJS;
    msg['text'] = text.toJS;
    _worker?.postMessage(msg);
  }

  /// Dispose the worker.
  void dispose() {
    _worker?.terminate();
    _worker = null;
  }

  static Future<String> _loadAssetAsString(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    return utf8.decode(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
  }

  static Future<Uint8List> _loadAssetBytes(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }

  void _handleMessage(web.MessageEvent event) {
    final data = event.data! as JSObject;
    final type = (data.getProperty('type'.toJS)! as JSString).toDart;

    if (type == 'ready') {
      onReady();
    } else if (type == 'result') {
      final result =
          (data.getProperty('result'.toJS)! as JSString).toDart;
      final elapsed =
          (data.getProperty('elapsed'.toJS)! as JSNumber).toDartDouble;
      onResult(result, elapsed);
    } else if (type == 'log') {
      final msg =
          (data.getProperty('message'.toJS)! as JSString).toDart;
      print('[punct-worker] $msg');
    } else if (type == 'error') {
      final msg =
          (data.getProperty('message'.toJS)! as JSString).toDart;
      onError(msg);
    }
  }
}
