// Web Worker support for TTS generation.
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;

import './generated_audio.dart';
import './model_web.dart' as m;
import './web_audio.dart' as web_audio;

typedef OnChunkCallback = void Function(AudioChunk chunk);
typedef OnDoneCallback = void Function(GeneratedAudioItem item);
typedef OnReadyCallback = void Function(int numSpeakers);
typedef OnErrorCallback = void Function(String message);

/// Manages a Web Worker for TTS generation.
class TtsWorker {
  web.Worker? _worker;
  final OnChunkCallback onChunk;
  final OnDoneCallback onDone;
  final OnReadyCallback onReady;
  final OnErrorCallback onError;

  String _pendingLabel = '';

  TtsWorker({
    required this.onChunk,
    required this.onDone,
    required this.onReady,
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
    _worker = web.Worker('tts-worker.js'.toJS);

    // Listen for messages from the worker.
    _worker!.onmessage = (web.MessageEvent event) {
      _handleMessage(event);
    }.toJS;

    // Handle worker startup failures (e.g. failed to load tts-worker.js).
    _worker!.onerror = (web.ErrorEvent event) {
      onError('Worker error: ${event.message}');
    }.toJS;

    // Load JS glue source, TTS helpers, and WASM binary from Flutter assets.
    final jsGlueSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.js');
    final ttsJsSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-tts.js');
    final wasmData = await _loadAssetBytes(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.wasm');

    // Build model files map.
    final jsModelFiles = JSObject();
    for (final entry in modelFiles.entries) {
      final bytes = Uint8List.fromList(entry.value);
      jsModelFiles[entry.key] = bytes.buffer.toJS;
    }

    // Convert OfflineTtsConfig to JS format for the worker.
    final jsConfig = m.configToJs(config);

    // Send init message with JS glue, TTS helpers, WASM binary, model files, and config.
    final initMsg = JSObject();
    initMsg['type'] = 'init'.toJS;
    initMsg['jsGlueSource'] = jsGlueSource.toJS;
    initMsg['ttsJsSource'] = ttsJsSource.toJS;
    initMsg['wasmBinary'] = wasmData.buffer.toJS;
    initMsg['modelFiles'] = jsModelFiles;
    initMsg['config'] = jsConfig;
    _worker!.postMessage(initMsg);
  }

  /// Start audio generation.
  void generate({
    required String text,
    int sid = 0,
    double speed = 1.0,
    int generationId = 0,
    Float32List? referenceAudio,
    int referenceSampleRate = 0,
    int numSteps = 5,
  }) {
    _pendingLabel = GeneratedAudioItem.makeLabel(text);
    final msg = JSObject();
    msg['type'] = 'generate'.toJS;
    msg['text'] = text.toJS;
    msg['sid'] = sid.toJS;
    msg['speed'] = speed.toJS;
    msg['generationId'] = generationId.toJS;
    msg['numSteps'] = numSteps.toJS;

    if (referenceAudio != null && referenceAudio.isNotEmpty) {
      msg['referenceAudio'] = referenceAudio.buffer.toJS;
      msg['referenceSampleRate'] = referenceSampleRate.toJS;
    }
    _worker?.postMessage(msg);
  }

  /// Cancel the current generation.
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

  /// Load a Flutter asset as a UTF-8 string.
  static Future<String> _loadAssetAsString(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    return utf8.decode(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
  }

  /// Load a Flutter asset as bytes.
  static Future<Uint8List> _loadAssetBytes(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }

  /// Convert a JS ArrayBuffer to a Dart ByteBuffer.
  static ByteBuffer _toByteBuffer(JSAny jsValue) {
    // Wrap in Uint8Array and use toDart to copy bytes.
    final uint8Ctor =
        globalContext.getProperty('Uint8Array'.toJS) as JSFunction;
    final view = uint8Ctor.callAsConstructor(jsValue) as JSUint8Array;
    return view.toDart.buffer;
  }

  void _handleMessage(web.MessageEvent event) {
    final data = event.data! as JSObject;
    final type = (data.getProperty('type'.toJS)! as JSString).toDart;

    if (type == 'ready') {
      final numSpeakers =
          (data.getProperty('numSpeakers'.toJS)! as JSNumber).toDartInt;
      onReady(numSpeakers);
    } else if (type == 'chunk') {
      final samplesBuffer = _toByteBuffer(data.getProperty('samples'.toJS)!);
      final samples = samplesBuffer.asFloat32List();
      final progress =
          (data.getProperty('progress'.toJS)! as JSNumber).toDartDouble;
      final sampleRate =
          (data.getProperty('sampleRate'.toJS)! as JSNumber).toDartInt;
      final genId =
          (data.getProperty('generationId'.toJS) as JSNumber?)?.toDartInt ?? 0;
      onChunk(AudioChunk(
        samples: Float32List.fromList(samples),
        progress: progress,
        sampleRate: sampleRate,
        generationId: genId,
      ));
    } else if (type == 'done') {
      final samplesBuffer = _toByteBuffer(data.getProperty('samples'.toJS)!);
      final samples = samplesBuffer.asFloat32List();
      final sampleRate =
          (data.getProperty('sampleRate'.toJS)! as JSNumber).toDartInt;
      final duration =
          (data.getProperty('duration'.toJS)! as JSNumber).toDartDouble;
      final elapsed =
          (data.getProperty('elapsed'.toJS)! as JSNumber).toDartDouble;
      final genId =
          (data.getProperty('generationId'.toJS) as JSNumber?)?.toDartInt ?? 0;

      final wavBytes =
          web_audio.encodeWav(Float32List.fromList(samples), sampleRate);
      onDone(GeneratedAudioItem(
        label: _pendingLabel,
        generationId: genId,
        wavBytes: wavBytes,
        duration: duration,
        elapsed: elapsed,
        sampleRate: sampleRate,
      ));
      _pendingLabel = '';
    } else if (type == 'log') {
      final msg =
          (data.getProperty('message'.toJS)! as JSString).toDart;
      print('[tts-worker] $msg');
    } else if (type == 'error') {
      final msg =
          (data.getProperty('message'.toJS)! as JSString).toDart;
      onError(msg);
    }
  }
}
