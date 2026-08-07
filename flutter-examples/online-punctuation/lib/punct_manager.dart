// Copyright (c)  2026  Xiaomi Corporation
//
// Punctuation manager — handles lifecycle for both native and web.
//
// On native: communicates with a background isolate via SendPort/ReceivePort.
// On web: delegates to PunctWorker which communicates with a Web Worker.

import 'dart:async';
import 'dart:isolate';

import 'package:flutter/foundation.dart' show kDebugMode, kIsWeb;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './model.dart' if (dart.library.js_interop) './model_web.dart' as m;
import './worker_web.dart' if (dart.library.io) './worker_stub.dart'
    as worker_lib;

/// State of the punctuation engine.
enum PunctState { uninitialized, initializing, initialized }

// ── Messages from main isolate → background isolate (native only) ────────

sealed class _ToWorker {}

class _PunctRequest extends _ToWorker {
  final String text;
  _PunctRequest(this.text);
}

class _DisposeRequest extends _ToWorker {}

// ── Messages from background isolate → main isolate (native only) ────────

sealed class _FromWorker {}

class _Ready extends _FromWorker {}

class _PunctDone extends _FromWorker {
  final String result;
  final double elapsed;
  _PunctDone(this.result, this.elapsed);
}

class _WorkerError extends _FromWorker {
  final String message;
  _WorkerError(this.message);
}

// ── PunctManager ─────────────────────────────────────────────────────────

class PunctManager {
  final _logController = StreamController<String>.broadcast();
  final _resultController = StreamController<PunctResult>.broadcast();

  Stream<String> get logStream => _logController.stream;
  Stream<PunctResult> get resultStream => _resultController.stream;

  // Native: isolate-based.
  Isolate? _isolate;
  SendPort? _sendPort;

  // Web: Web Worker-based.
  worker_lib.PunctWorker? _worker;

  PunctState _state = PunctState.uninitialized;

  PunctState get state => _state;
  bool get isInitialized => _state == PunctState.initialized;

  /// Initialize the punctuation engine.
  Future<void> init() async {
    if (_state != PunctState.uninitialized) return;
    _state = PunctState.initializing;

    if (kIsWeb) {
      await _initWeb();
    } else {
      await _initNative();
    }
  }

  /// Add punctuation to text.
  void punctuate(String text) {
    if (_state != PunctState.initialized) {
      _logController.add('Error: Punctuation not initialized');
      return;
    }

    if (kDebugMode) {
      print('[punct_manager] punctuate: text="$text"');
    }

    if (kIsWeb) {
      _worker?.punctuate(text: text);
    } else {
      _sendPort!.send(_PunctRequest(text));
    }
  }

  /// Dispose the engine and release resources.
  void dispose() {
    _state = PunctState.uninitialized;
    if (kIsWeb) {
      _worker?.dispose();
      _worker = null;
    } else {
      _sendPort?.send(_DisposeRequest());
      _isolate?.kill(priority: Isolate.immediate);
      _isolate = null;
      _sendPort = null;
    }
    _logController.close();
    _resultController.close();
  }

  // ── Web (Web Worker) ──────────────────────────────────────────────────

  Future<void> _initWeb() async {
    try {
      final readyCompleter = Completer<void>();

      _worker = worker_lib.PunctWorker(
        onReady: () {
          _state = PunctState.initialized;
          _logController.add('Punctuation ready');
          if (!readyCompleter.isCompleted) readyCompleter.complete();
        },
        onResult: (result, elapsed) {
          _resultController.add(PunctResult(result: result, elapsed: elapsed));
        },
        onError: (msg) {
          _state = PunctState.uninitialized;
          _logController.add('Error: $msg');
          if (!readyCompleter.isCompleted) readyCompleter.completeError(msg);
        },
      );

      await _worker!.init();
      await readyCompleter.future;
    } catch (e) {
      _state = PunctState.uninitialized;
      _logController.add('Error: $e');
    }
  }

  // ── Native (isolate-based) ──────────────────────────────────────────────

  Future<void> _initNative() async {
    try {
      _logController.add('Preparing model...');
      final config = await m.prepareModelConfig();

      if (kDebugMode) {
        print('[punct_manager] config: $config');
      }

      sherpa_onnx.initBindings();

      _logController.add('Starting punctuation isolate...');
      final receivePort = ReceivePort();
      _isolate = await Isolate.spawn(_workerEntry, receivePort.sendPort);

      final readyCompleter = Completer<void>();

      receivePort.listen((message) {
        if (message is SendPort) {
          _sendPort = message;
          message.send(config);
        } else if (message is _Ready) {
          _state = PunctState.initialized;
          _logController.add('Punctuation ready');
          if (!readyCompleter.isCompleted) readyCompleter.complete();
        } else if (message is _PunctDone) {
          _resultController.add(
              PunctResult(result: message.result, elapsed: message.elapsed));
        } else if (message is _WorkerError) {
          _state = PunctState.uninitialized;
          _logController.add('Error: ${message.message}');
          if (!readyCompleter.isCompleted) {
            readyCompleter.completeError(message.message);
          }
        }
      });

      return readyCompleter.future;
    } catch (e) {
      _state = PunctState.uninitialized;
      _logController.add('Error: $e');
      rethrow;
    }
  }

  static void _workerEntry(SendPort mainSendPort) {
    final receivePort = ReceivePort();
    mainSendPort.send(receivePort.sendPort);

    sherpa_onnx.OnlinePunctuation? punct;

    receivePort.listen((message) {
      if (message is sherpa_onnx.OnlinePunctuationConfig) {
        try {
          sherpa_onnx.initBindings();
          punct = sherpa_onnx.OnlinePunctuation(config: message);
          mainSendPort.send(_Ready());
        } catch (e) {
          mainSendPort.send(_WorkerError('$e'));
        }
      } else if (message is _PunctRequest && punct != null) {
        try {
          final stopwatch = Stopwatch()..start();
          final result = punct!.addPunct(message.text);
          stopwatch.stop();
          final elapsed = stopwatch.elapsedMilliseconds / 1000.0;
          mainSendPort.send(_PunctDone(result, elapsed));
        } catch (e) {
          mainSendPort.send(_WorkerError('$e'));
        }
      } else if (message is _DisposeRequest) {
        punct?.free();
        punct = null;
        receivePort.close();
      }
    });
  }
}

/// Result of a punctuation operation.
class PunctResult {
  final String result;
  final double elapsed;
  PunctResult({required this.result, required this.elapsed});
}
