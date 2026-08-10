// Web Worker support for VAD.
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;

import './model_web.dart' as m;
import './vad_manager.dart' show VadSegment;

typedef OnReadyCallback = void Function();
typedef OnProgressCallback = void Function(double progress);
typedef OnResultCallback = void Function(
    List<VadSegment> segments, double elapsed, double audioDuration);
typedef OnErrorCallback = void Function(String message);

/// Manages a Web Worker for VAD.
class VadWorker {
  web.Worker? _worker;
  final OnReadyCallback onReady;
  final OnProgressCallback onProgress;
  final OnResultCallback onResult;
  final OnErrorCallback onError;

  VadWorker({
    required this.onReady,
    required this.onProgress,
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
    _worker = web.Worker('vad-worker.js'.toJS);

    // Listen for messages from the worker.
    _worker!.onmessage = (web.MessageEvent event) {
      _handleMessage(event);
    }.toJS;

    // Handle worker startup failures.
    _worker!.onerror = (web.ErrorEvent event) {
      onError('Worker error: ${event.message}');
    }.toJS;

    // Load JS glue source, VAD helpers, and WASM binary.
    final jsGlueSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.js');
    final vadJsSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-vad.js');
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
    initMsg['vadJsSource'] = vadJsSource.toJS;
    initMsg['wasmBinary'] = wasmData.buffer.toJS;
    initMsg['modelFiles'] = jsModelFiles;
    initMsg['config'] = jsConfig;
    _worker!.postMessage(initMsg);
  }

  /// Run VAD on audio samples.
  void runVad({
    required Float32List samples,
    required int sampleRate,
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) {
    final msg = JSObject();
    msg['type'] = 'runVad'.toJS;
    msg['samples'] = samples.buffer.toJS;
    msg['sampleRate'] = sampleRate.toJS;
    msg['threshold'] = threshold.toJS;
    msg['minSilenceDuration'] = minSilenceDuration.toJS;
    msg['minSpeechDuration'] = minSpeechDuration.toJS;
    msg['maxSpeechDuration'] = maxSpeechDuration.toJS;
    _worker?.postMessage(msg);
  }

  /// Cancel the current VAD processing.
  void cancel() {
    final msg = JSObject();
    msg['type'] = 'cancel'.toJS;
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
    } else if (type == 'progress') {
      final progress =
          (data.getProperty('progress'.toJS)! as JSNumber).toDartDouble;
      onProgress(progress);
    } else if (type == 'result') {
      final segmentsJs =
          data.getProperty('segments'.toJS)! as JSArray;
      final elapsed =
          (data.getProperty('elapsed'.toJS)! as JSNumber).toDartDouble;
      final audioDuration =
          (data.getProperty('audioDuration'.toJS)! as JSNumber).toDartDouble;

      final segments = <VadSegment>[];
      for (int i = 0; i < segmentsJs.length; i++) {
        final seg = segmentsJs[i] as JSObject;
        final start =
            (seg.getProperty('start'.toJS)! as JSNumber).toDartDouble;
        final end =
            (seg.getProperty('end'.toJS)! as JSNumber).toDartDouble;
        final samplesJs =
            seg.getProperty('samples'.toJS)! as JSFloat32Array;
        segments.add(VadSegment(
          start: start,
          end: end,
          samples: Float32List.fromList(samplesJs.toDart),
        ));
      }
      onResult(segments, elapsed, audioDuration);
    } else if (type == 'log') {
      final msg =
          (data.getProperty('message'.toJS)! as JSString).toDart;
      print('[vad-worker] $msg');
    } else if (type == 'error') {
      final msg =
          (data.getProperty('message'.toJS)! as JSString).toDart;
      onError(msg);
    }
  }
}
