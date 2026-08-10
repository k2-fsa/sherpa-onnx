// Copyright (c)  2026  Xiaomi Corporation
//
// VAD manager for real-time microphone input.
//
// On native: communicates with a background isolate via SendPort/ReceivePort.
// On web: delegates to VadWorker which communicates with a Web Worker.

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kDebugMode, kIsWeb;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './model.dart' if (dart.library.js_interop) './model_web.dart' as m;
import './worker_web.dart' if (dart.library.io) './worker_stub.dart'
    as worker_lib;

/// State of the VAD engine.
enum VadState { uninitialized, initializing, initialized }

// ── Messages from main isolate → background isolate (native only) ────────

sealed class _ToWorker {}

class _ConfigRequest extends _ToWorker {
  final sherpa_onnx.VadModelConfig config;
  _ConfigRequest(this.config);
}

class _AudioChunkRequest extends _ToWorker {
  final Float32List samples;
  _AudioChunkRequest(this.samples);
}

class _VadRunRequest extends _ToWorker {
  final Float32List samples;
  final int sampleRate;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final double maxSpeechDuration;
  _VadRunRequest({
    required this.samples,
    required this.sampleRate,
    required this.threshold,
    required this.minSilenceDuration,
    required this.minSpeechDuration,
    required this.maxSpeechDuration,
  });
}

class _DisposeRequest extends _ToWorker {}

// ── Messages from background isolate → main isolate (native only) ────────

sealed class _FromWorker {}

class _Ready extends _FromWorker {}

class _SpeechStateChanged extends _FromWorker {
  final bool isSpeaking;
  _SpeechStateChanged(this.isSpeaking);
}

class _SegmentCountChanged extends _FromWorker {
  final int count;
  _SegmentCountChanged(this.count);
}

class _SegmentDetected extends _FromWorker {
  final double start;
  final double end;
  final Float32List samples;
  _SegmentDetected(this.start, this.end, this.samples);
}

class _WorkerError extends _FromWorker {
  final String message;
  _WorkerError(this.message);
}

class _VadDone extends _FromWorker {
  final List<VadSegment> segments;
  final double elapsed;
  final double audioDuration;
  _VadDone(this.segments, this.elapsed, this.audioDuration);
}

/// A detected speech segment.
class VadSegment {
  final double start;
  final double end;
  final Float32List samples;
  VadSegment({required this.start, required this.end, required this.samples});
}

/// Result of a VAD processing run.
class VadResult {
  final List<VadSegment> segments;
  final double elapsed;
  final double audioDuration;
  VadResult({
    required this.segments,
    required this.elapsed,
    required this.audioDuration,
  });
}

// ── VadMicManager ────────────────────────────────────────────────────────

class VadMicManager {
  final _logController = StreamController<String>.broadcast();
  final _speechController = StreamController<bool>.broadcast();
  final _segmentCountController = StreamController<int>.broadcast();
  final _segmentsController = StreamController<VadSegment>.broadcast();
  final _resultController = StreamController<VadResult>.broadcast();

  Stream<String> get logStream => _logController.stream;
  Stream<bool> get speechStream => _speechController.stream;
  Stream<int> get segmentCountStream => _segmentCountController.stream;
  Stream<VadSegment> get segmentsStream => _segmentsController.stream;
  Stream<VadResult> get resultStream => _resultController.stream;

  // Native: isolate-based.
  Isolate? _isolate;
  SendPort? _sendPort;

  // Web: Web Worker-based.
  worker_lib.VadWorker? _worker;

  VadState _state = VadState.uninitialized;
  int _segmentCount = 0;

  VadState get state => _state;
  int get segmentCount => _segmentCount;

  /// Initialize the VAD engine.
  Future<void> init({
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) async {
    if (_state != VadState.uninitialized) return;
    _state = VadState.initializing;

    if (kIsWeb) {
      await _initWeb(
        threshold: threshold,
        minSilenceDuration: minSilenceDuration,
        minSpeechDuration: minSpeechDuration,
        maxSpeechDuration: maxSpeechDuration,
      );
    } else {
      await _initNative(
        threshold: threshold,
        minSilenceDuration: minSilenceDuration,
        minSpeechDuration: minSpeechDuration,
        maxSpeechDuration: maxSpeechDuration,
      );
    }
  }

  /// Send audio samples to the VAD engine.
  void acceptWaveform(Float32List samples) {
    if (_state != VadState.initialized) return;

    if (kIsWeb) {
      _worker?.acceptWaveform(samples);
    } else {
      _sendPort!.send(_AudioChunkRequest(samples));
    }
  }

  /// Dispose the engine and release resources.
  void dispose() {
    _state = VadState.uninitialized;
    _segmentCount = 0;
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
    _speechController.close();
    _segmentCountController.close();
    _segmentsController.close();
    _resultController.close();
  }

  // ── Web (Web Worker) ──────────────────────────────────────────────────

  Future<void> _initWeb({
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) async {
    try {
      final readyCompleter = Completer<void>();

      _worker = worker_lib.VadWorker(
        onReady: () {
          _state = VadState.initialized;
          _logController.add('VAD ready');
          if (!readyCompleter.isCompleted) readyCompleter.complete();
        },
        onSpeechStateChanged: (isSpeaking) {
          _speechController.add(isSpeaking);
        },
        onSegmentCountChanged: (count) {
          _segmentCount = count;
          _segmentCountController.add(count);
        },
        onSegmentDetected: (seg) {
          _segmentsController.add(seg);
        },
        onError: (msg) {
          _state = VadState.uninitialized;
          _logController.add('Error: $msg');
          if (!readyCompleter.isCompleted) readyCompleter.completeError(msg);
        },
      );

      await _worker!.init(
        threshold: threshold,
        minSilenceDuration: minSilenceDuration,
        minSpeechDuration: minSpeechDuration,
        maxSpeechDuration: maxSpeechDuration,
      );
      await readyCompleter.future;
    } catch (e) {
      _state = VadState.uninitialized;
      _logController.add('Error: $e');
    }
  }

  // ── Native (isolate-based) ──────────────────────────────────────────────

  Future<void> _initNative({
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) async {
    try {
      _logController.add('Preparing model...');
      final baseConfig = await m.prepareModelConfig();

      // Apply user-configured parameters based on which model is active.
      final isTenVad = baseConfig.tenVad.model.isNotEmpty;
      final config = sherpa_onnx.VadModelConfig(
        sileroVad: isTenVad
            ? baseConfig.sileroVad
            : sherpa_onnx.SileroVadModelConfig(
                model: baseConfig.sileroVad.model,
                threshold: threshold,
                minSilenceDuration: minSilenceDuration,
                minSpeechDuration: minSpeechDuration,
                windowSize: baseConfig.sileroVad.windowSize,
                maxSpeechDuration: maxSpeechDuration,
              ),
        tenVad: isTenVad
            ? sherpa_onnx.TenVadModelConfig(
                model: baseConfig.tenVad.model,
                threshold: threshold,
                minSilenceDuration: minSilenceDuration,
                minSpeechDuration: minSpeechDuration,
                windowSize: baseConfig.tenVad.windowSize,
                maxSpeechDuration: maxSpeechDuration,
              )
            : baseConfig.tenVad,
        sampleRate: baseConfig.sampleRate,
        numThreads: baseConfig.numThreads,
        provider: baseConfig.provider,
        debug: baseConfig.debug,
      );

      sherpa_onnx.initBindings();

      _logController.add('Starting VAD isolate...');
      final receivePort = ReceivePort();
      _isolate = await Isolate.spawn(_workerEntry, receivePort.sendPort);

      final readyCompleter = Completer<void>();

      receivePort.listen((message) {
        if (message is SendPort) {
          _sendPort = message;
          message.send(config);
        } else if (message is _Ready) {
          _state = VadState.initialized;
          _logController.add('VAD ready');
          if (!readyCompleter.isCompleted) readyCompleter.complete();
        } else if (message is _SpeechStateChanged) {
          _speechController.add(message.isSpeaking);
        } else if (message is _SegmentCountChanged) {
          _segmentCount = message.count;
          _segmentCountController.add(message.count);
        } else if (message is _SegmentDetected) {
          _segmentsController.add(VadSegment(
            start: message.start,
            end: message.end,
            samples: message.samples,
          ));
        } else if (message is _WorkerError) {
          _state = VadState.uninitialized;
          _logController.add('Error: ${message.message}');
          if (!readyCompleter.isCompleted) {
            readyCompleter.completeError(message.message);
          }
        }
      });

      return readyCompleter.future;
    } catch (e) {
      _state = VadState.uninitialized;
      _logController.add('Error: $e');
      rethrow;
    }
  }

  static void _workerEntry(SendPort mainSendPort) {
    final receivePort = ReceivePort();
    mainSendPort.send(receivePort.sendPort);

    sherpa_onnx.VoiceActivityDetector? vad;
    int segmentCount = 0;
    bool isSpeaking = false;

    receivePort.listen((message) {
      if (message is sherpa_onnx.VadModelConfig) {
        try {
          sherpa_onnx.initBindings();
          vad = sherpa_onnx.VoiceActivityDetector(
              config: message, bufferSizeInSeconds: 30);
          mainSendPort.send(_Ready());
        } catch (e) {
          mainSendPort.send(_WorkerError('$e'));
        }
      } else if (message is _AudioChunkRequest && vad != null) {
        try {
          vad!.acceptWaveform(message.samples);

          // Update speech state.
          final detected = vad!.isDetected();
          if (detected != isSpeaking) {
            isSpeaking = detected;
            mainSendPort.send(_SpeechStateChanged(isSpeaking));
          }

          // Collect completed segments (available after speech ends).
          while (!vad!.isEmpty()) {
            final seg = vad!.front();
            final sampleRate = 16000; // Always 16kHz for VAD.
            final startSec = seg.start / sampleRate;
            final endSec = startSec + seg.samples.length / sampleRate;
            mainSendPort.send(_SegmentDetected(startSec, endSec, seg.samples));
            vad!.pop();
            segmentCount++;
          }
          if (segmentCount > 0) {
            mainSendPort.send(_SegmentCountChanged(segmentCount));
          }
        } catch (e) {
          mainSendPort.send(_WorkerError('$e'));
        }
      } else if (message is _DisposeRequest) {
        vad?.free();
        vad = null;
        receivePort.close();
      }
    });
  }
}
