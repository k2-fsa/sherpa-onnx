// Web Worker support for real-time VAD.
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
typedef OnSpeechStateChangedCallback = void Function(bool isSpeaking);
typedef OnSegmentCountChangedCallback = void Function(int count);
typedef OnSegmentDetectedCallback = void Function(VadSegment segment);
typedef OnErrorCallback = void Function(String message);

/// Manages a Web Worker for real-time VAD.
class VadWorker {
  web.Worker? _worker;
  final OnReadyCallback onReady;
  final OnSpeechStateChangedCallback onSpeechStateChanged;
  final OnSegmentCountChangedCallback onSegmentCountChanged;
  final OnSegmentDetectedCallback onSegmentDetected;
  final OnErrorCallback onError;

  VadWorker({
    required this.onReady,
    required this.onSpeechStateChanged,
    required this.onSegmentCountChanged,
    required this.onSegmentDetected,
    required this.onError,
  });

  /// Initialize the worker.
  Future<void> init({
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) async {
    final modelFiles = await m.loadModelFileBytes();
    final config = await m.prepareModelConfig();

    _worker = web.Worker('vad-worker.js'.toJS);

    _worker!.onmessage = (web.MessageEvent event) {
      _handleMessage(event);
    }.toJS;

    _worker!.onerror = (web.ErrorEvent event) {
      onError('Worker error: ${event.message}');
    }.toJS;

    final jsGlueSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.js');
    final vadJsSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-vad.js');
    final wasmData = await _loadAssetBytes(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.wasm');

    final jsModelFiles = JSObject();
    for (final entry in modelFiles.entries) {
      final bytes = Uint8List.fromList(entry.value);
      jsModelFiles[entry.key] = bytes.buffer.toJS;
    }

    final jsConfig = m.configToJs(config);
    // Override with user parameters.
    final sileroVad = jsConfig.getProperty('sileroVad'.toJS) as JSObject?;
    if (sileroVad != null) {
      sileroVad['threshold'] = threshold.toJS;
      sileroVad['minSilenceDuration'] = minSilenceDuration.toJS;
      sileroVad['minSpeechDuration'] = minSpeechDuration.toJS;
      sileroVad['maxSpeechDuration'] = maxSpeechDuration.toJS;
    }

    final initMsg = JSObject();
    initMsg['type'] = 'init'.toJS;
    initMsg['jsGlueSource'] = jsGlueSource.toJS;
    initMsg['vadJsSource'] = vadJsSource.toJS;
    initMsg['wasmBinary'] = wasmData.buffer.toJS;
    initMsg['modelFiles'] = jsModelFiles;
    initMsg['config'] = jsConfig;
    _worker!.postMessage(initMsg);
  }

  /// Send audio samples to the worker.
  void acceptWaveform(Float32List samples) {
    final msg = JSObject();
    msg['type'] = 'acceptWaveform'.toJS;
    msg['samples'] = samples.buffer.toJS;
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
    } else if (type == 'speechStateChanged') {
      final isSpeaking =
          (data.getProperty('isSpeaking'.toJS)! as JSBoolean).toDart;
      onSpeechStateChanged(isSpeaking);
    } else if (type == 'segmentCountChanged') {
      final count =
          (data.getProperty('count'.toJS)! as JSNumber).toDartInt;
      onSegmentCountChanged(count);
    } else if (type == 'segmentDetected') {
      final start =
          (data.getProperty('start'.toJS)! as JSNumber).toDartDouble;
      final end =
          (data.getProperty('end'.toJS)! as JSNumber).toDartDouble;
      final samplesBuffer =
          data.getProperty('samples'.toJS)! as JSArrayBuffer;
      final samplesArray = JSFloat32Array.new(samplesBuffer);
      onSegmentDetected(VadSegment(
        start: start,
        end: end,
        samples: Float32List.fromList(samplesArray.toDart),
      ));
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
