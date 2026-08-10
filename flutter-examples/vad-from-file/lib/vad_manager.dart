// Copyright (c)  2026  Xiaomi Corporation
//
// VAD manager — handles VAD lifecycle for both native and web.
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

class _RunVadRequest extends _ToWorker {
  final Float32List samples;
  final int sampleRate;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final double maxSpeechDuration;
  _RunVadRequest({
    required this.samples,
    required this.sampleRate,
    required this.threshold,
    required this.minSilenceDuration,
    required this.minSpeechDuration,
    required this.maxSpeechDuration,
  });
}

class _UpdateConfigRequest extends _ToWorker {
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final double maxSpeechDuration;
  _UpdateConfigRequest({
    required this.threshold,
    required this.minSilenceDuration,
    required this.minSpeechDuration,
    required this.maxSpeechDuration,
  });
}

class _CancelRequest extends _ToWorker {}

class _DisposeRequest extends _ToWorker {}

// ── Messages from background isolate → main isolate (native only) ────────

sealed class _FromWorker {}

class _Ready extends _FromWorker {}

class _ProgressUpdate extends _FromWorker {
  final double progress;
  _ProgressUpdate(this.progress);
}

class _VadDone extends _FromWorker {
  final List<VadSegment> segments;
  final double elapsed;
  final double audioDuration;
  _VadDone(this.segments, this.elapsed, this.audioDuration);
}

class _WorkerError extends _FromWorker {
  final String message;
  _WorkerError(this.message);
}

// ── VadSegment ───────────────────────────────────────────────────────────

/// A detected speech segment with start/end times.
class VadSegment {
  final double start;
  final double end;
  final Float32List samples;
  VadSegment({required this.start, required this.end, required this.samples});
}

// ── VadResult ────────────────────────────────────────────────────────────

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

// ── VadManager ───────────────────────────────────────────────────────────

class VadManager {
  final _logController = StreamController<String>.broadcast();
  final _progressController = StreamController<double>.broadcast();
  final _resultController = StreamController<VadResult>.broadcast();

  Stream<String> get logStream => _logController.stream;
  Stream<double> get progressStream => _progressController.stream;
  Stream<VadResult> get resultStream => _resultController.stream;

  // Native: isolate-based.
  Isolate? _isolate;
  SendPort? _sendPort;

  // Web: Web Worker-based.
  worker_lib.VadWorker? _worker;

  VadState _state = VadState.uninitialized;

  VadState get state => _state;
  bool get isInitialized => _state == VadState.initialized;

  /// Initialize the VAD engine.
  Future<void> init() async {
    if (_state != VadState.uninitialized) return;
    _state = VadState.initializing;

    if (kIsWeb) {
      await _initWeb();
    } else {
      await _initNative();
    }
  }

  /// Run VAD on the given audio samples.
  void runVad({
    required Float32List samples,
    required int sampleRate,
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) {
    if (_state != VadState.initialized) {
      _logController.add('Error: VAD not initialized');
      return;
    }

    if (kIsWeb) {
      _worker?.runVad(
        samples: samples,
        sampleRate: sampleRate,
        threshold: threshold,
        minSilenceDuration: minSilenceDuration,
        minSpeechDuration: minSpeechDuration,
        maxSpeechDuration: maxSpeechDuration,
      );
    } else {
      _sendPort!.send(_RunVadRequest(
        samples: samples,
        sampleRate: sampleRate,
        threshold: threshold,
        minSilenceDuration: minSilenceDuration,
        minSpeechDuration: minSpeechDuration,
        maxSpeechDuration: maxSpeechDuration,
      ));
    }
  }

  /// Cancel the current VAD processing.
  void cancel() {
    if (kIsWeb) {
      _worker?.cancel();
    } else {
      _sendPort?.send(_CancelRequest());
    }
  }

  /// Dispose the engine and release resources.
  void dispose() {
    _state = VadState.uninitialized;
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
    _progressController.close();
    _resultController.close();
  }

  // ── Web (Web Worker) ──────────────────────────────────────────────────

  Future<void> _initWeb() async {
    try {
      final readyCompleter = Completer<void>();

      _worker = worker_lib.VadWorker(
        onReady: () {
          _state = VadState.initialized;
          _logController.add('VAD ready');
          if (!readyCompleter.isCompleted) readyCompleter.complete();
        },
        onProgress: (progress) {
          _progressController.add(progress);
        },
        onResult: (segments, elapsed, audioDuration) {
          _resultController.add(VadResult(
            segments: segments,
            elapsed: elapsed,
            audioDuration: audioDuration,
          ));
        },
        onError: (msg) {
          _state = VadState.uninitialized;
          _logController.add('Error: $msg');
          if (!readyCompleter.isCompleted) readyCompleter.completeError(msg);
        },
      );

      await _worker!.init();
      await readyCompleter.future;
    } catch (e) {
      _state = VadState.uninitialized;
      _logController.add('Error: $e');
    }
  }

  // ── Native (isolate-based) ──────────────────────────────────────────────

  Future<void> _initNative() async {
    try {
      _logController.add('Preparing model...');
      final config = await m.prepareModelConfig();

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
        } else if (message is _ProgressUpdate) {
          _progressController.add(message.progress);
        } else if (message is _VadDone) {
          _resultController.add(VadResult(
            segments: message.segments,
            elapsed: message.elapsed,
            audioDuration: message.audioDuration,
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
    sherpa_onnx.VadModelConfig? vadConfig;
    bool cancelled = false;

    receivePort.listen((message) {
      if (message is sherpa_onnx.VadModelConfig) {
        try {
          sherpa_onnx.initBindings();
          vadConfig = message;
          vad = sherpa_onnx.VoiceActivityDetector(
              config: message, bufferSizeInSeconds: 30);
          mainSendPort.send(_Ready());
        } catch (e) {
          mainSendPort.send(_WorkerError('$e'));
        }
      } else if (message is _CancelRequest) {
        cancelled = true;
      } else if (message is _RunVadRequest && vadConfig != null) {
        try {
          cancelled = false;

          // Re-create VAD with the requested parameters.
          vad?.free();

          // Determine which model is active and apply user parameters.
          final isTenVad = vadConfig!.tenVad.model.isNotEmpty;
          final runConfig = sherpa_onnx.VadModelConfig(
            sileroVad: isTenVad
                ? vadConfig!.sileroVad
                : sherpa_onnx.SileroVadModelConfig(
                    model: vadConfig!.sileroVad.model,
                    threshold: message.threshold,
                    minSilenceDuration: message.minSilenceDuration,
                    minSpeechDuration: message.minSpeechDuration,
                    windowSize: vadConfig!.sileroVad.windowSize,
                    maxSpeechDuration: message.maxSpeechDuration,
                  ),
            tenVad: isTenVad
                ? sherpa_onnx.TenVadModelConfig(
                    model: vadConfig!.tenVad.model,
                    threshold: message.threshold,
                    minSilenceDuration: message.minSilenceDuration,
                    minSpeechDuration: message.minSpeechDuration,
                    windowSize: vadConfig!.tenVad.windowSize,
                    maxSpeechDuration: message.maxSpeechDuration,
                  )
                : vadConfig!.tenVad,
            sampleRate: vadConfig!.sampleRate,
            numThreads: vadConfig!.numThreads,
            provider: vadConfig!.provider,
            debug: vadConfig!.debug,
          );
          vad = sherpa_onnx.VoiceActivityDetector(
              config: runConfig, bufferSizeInSeconds: 30);

          final stopwatch = Stopwatch()..start();
          final windowSize = isTenVad
              ? vadConfig!.tenVad.windowSize
              : vadConfig!.sileroVad.windowSize;
          final sampleRate = message.sampleRate;
          final samples = message.samples;
          final numSamples = samples.length;
          final numIter = numSamples ~/ windowSize;
          final audioDuration = numSamples / sampleRate;

          final segments = <VadSegment>[];

          for (int i = 0; i < numIter; i++) {
            if (cancelled) break;

            final start = i * windowSize;
            final chunk = Float32List.sublistView(samples, start, start + windowSize);
            vad!.acceptWaveform(chunk);

            if (vad!.isDetected()) {
              while (!vad!.isEmpty()) {
                final seg = vad!.front();
                final segStart = seg.start / sampleRate;
                final segEnd = segStart + seg.samples.length / sampleRate;
                segments.add(VadSegment(
                  start: segStart,
                  end: segEnd,
                  samples: seg.samples,
                ));
                vad!.pop();
              }
            }

            // Report progress.
            final progress = (i + 1) / numIter;
            mainSendPort.send(_ProgressUpdate(progress));
          }

          vad!.flush();
          while (!vad!.isEmpty()) {
            final seg = vad!.front();
            final segStart = seg.start / sampleRate;
            final segEnd = segStart + seg.samples.length / sampleRate;
            segments.add(VadSegment(
              start: segStart,
              end: segEnd,
              samples: seg.samples,
            ));
            vad!.pop();
          }

          stopwatch.stop();
          final elapsed = stopwatch.elapsedMilliseconds / 1000.0;
          mainSendPort.send(_VadDone(segments, elapsed, audioDuration));
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
