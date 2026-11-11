// Copyright (c)  2026  Xiaomi Corporation
// Web implementation of VAD+ASR using a Web Worker.
import 'dart:async';
import 'dart:typed_data';

import './worker_web.dart';

/// A VAD segment with ASR transcription.
class VadAsrSegment {
  final double start;
  final double end;
  final Float32List samples;
  String text;

  VadAsrSegment({
    required this.start,
    required this.end,
    required this.samples,
    this.text = '',
  });
}

class VadAsrResult {
  final List<VadAsrSegment> segments;
  final double elapsed;
  final double audioDuration;
  const VadAsrResult({
    required this.segments,
    required this.elapsed,
    required this.audioDuration,
  });
}

enum VadAsrState { uninitialized, initializing, initialized }

class VadAsrManager {
  VadAsrState _state = VadAsrState.uninitialized;
  VadAsrState get state => _state;

  final _logController = StreamController<String>.broadcast();
  final _progressController = StreamController<double>.broadcast();
  final _segmentController = StreamController<VadAsrSegment>.broadcast();
  final _resultController = StreamController<VadAsrResult>.broadcast();

  Stream<String> get logStream => _logController.stream;
  Stream<double> get progressStream => _progressController.stream;
  Stream<VadAsrSegment> get segmentStream => _segmentController.stream;
  Stream<VadAsrResult> get resultStream => _resultController.stream;

  VadAsrWorker? _worker;
  bool _cancelled = false;
  int _runId = 0; // Incremented each run; stale results are dropped.

  Future<void> init({required String modelDir, required String vadModelDir}) async {
    if (_state != VadAsrState.uninitialized) return;
    _state = VadAsrState.initializing;
    _logController.add('Initializing VAD + ASR (web)...');

    try {
      await _initWeb();
    } catch (e) {
      _state = VadAsrState.uninitialized;
      _logController.add('Init error: $e');
      rethrow;
    }
  }

  /// Create (or recreate) the web worker and wait for it to be ready.
  Future<void> _initWeb() async {
    final readyCompleter = Completer<void>();

    _worker = VadAsrWorker(
      onReady: () {
        _state = VadAsrState.initialized;
        _logController.add('VAD + ASR ready');
        if (!readyCompleter.isCompleted) readyCompleter.complete();
      },
      onStarted: (runId) {
        if (runId == _runId) {
          _cancelled = false;
        }
      },
      onProgress: (progress) {
        if (!_cancelled) _progressController.add(progress);
      },
      onSegment: (seg) {
        if (!_cancelled && seg.text.trim().isNotEmpty) {
          _segmentController.add(seg);
        }
      },
      onResult: (segments, elapsed, audioDuration) {
        if (_cancelled) return;
        final filtered = segments.where((s) => s.text.trim().isNotEmpty).toList();
        _resultController.add(VadAsrResult(
          segments: List.unmodifiable(filtered),
          elapsed: elapsed,
          audioDuration: audioDuration,
        ));
      },
      onError: (msg) {
        _state = VadAsrState.uninitialized;
        _logController.add('Error: $msg');
        if (!readyCompleter.isCompleted) readyCompleter.completeError(msg);
      },
    );

    await _worker!.init();
    await readyCompleter.future.timeout(
      const Duration(seconds: 60),
      onTimeout: () => throw TimeoutException('Web worker init timed out'),
    );
  }

  void runVad({
    required Float32List samples,
    required int sampleRate,
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) {
    if (_state != VadAsrState.initialized) {
      _logController.add('Error: not initialized');
      return;
    }

    // If cancelled, terminate old worker and reinitialize.
    if (_cancelled) {
      _worker?.dispose();
      _worker = null;
      _cancelled = false;
      _initWeb().then((_) {
        _runId++;
        _worker?.runVad(
          runId: _runId,
          samples: samples,
          sampleRate: sampleRate,
          threshold: threshold,
          minSilenceDuration: minSilenceDuration,
          minSpeechDuration: minSpeechDuration,
          maxSpeechDuration: maxSpeechDuration,
        );
      });
      return;
    }

    _runId++;
    _worker?.runVad(
      runId: _runId,
      samples: samples,
      sampleRate: sampleRate,
      threshold: threshold,
      minSilenceDuration: minSilenceDuration,
      minSpeechDuration: minSpeechDuration,
      maxSpeechDuration: maxSpeechDuration,
    );
  }

  void cancel() {
    _cancelled = true;
    _worker?.cancel();
  }

  void dispose() {
    _worker?.dispose();
    _worker = null;
    _state = VadAsrState.uninitialized;
    _logController.close();
    _progressController.close();
    _segmentController.close();
    _resultController.close();
  }
}
