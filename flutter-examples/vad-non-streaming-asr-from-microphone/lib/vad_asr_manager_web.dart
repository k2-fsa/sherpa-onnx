// Copyright (c)  2026  Xiaomi Corporation
// Web implementation of VAD+ASR using a Web Worker.
import 'dart:async';
import 'dart:typed_data';

import './worker_web.dart';

/// A VAD segment with ASR transcription.
class VadAsrSegment {
  final int index;
  final double start; // seconds
  final double end; // seconds
  final Float32List samples;
  String text;
  double elapsedSeconds;

  VadAsrSegment({
    required this.index,
    required this.start,
    required this.end,
    required this.samples,
    this.text = '',
    this.elapsedSeconds = 0,
  });
}

enum VadAsrState { uninitialized, initializing, initialized }

/// ASR text update for a segment.
class TextUpdate {
  final int index;
  final String text;
  final double elapsedSeconds;
  const TextUpdate(this.index, this.text, this.elapsedSeconds);
}

class VadAsrManager {
  VadAsrState _state = VadAsrState.uninitialized;
  VadAsrState get state => _state;

  final _logController = StreamController<String>.broadcast();
  final _segmentController = StreamController<VadAsrSegment>.broadcast();
  final _textUpdateController = StreamController<TextUpdate>.broadcast();
  final _speechController = StreamController<bool>.broadcast();

  Stream<String> get logStream => _logController.stream;
  Stream<VadAsrSegment> get segmentStream => _segmentController.stream;
  Stream<TextUpdate> get textUpdateStream => _textUpdateController.stream;
  Stream<bool> get speechStream => _speechController.stream;

  VadAsrMicWorker? _worker;

  Future<void> init({
    required String modelDir,
    required String vadModelDir,
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) async {
    if (_state != VadAsrState.uninitialized) return;
    _state = VadAsrState.initializing;
    _logController.add('Initializing VAD + ASR (web)...');

    try {
      final readyCompleter = Completer<void>();

      _worker = VadAsrMicWorker(
        onReady: () {
          _state = VadAsrState.initialized;
          _logController.add('VAD + ASR ready (web)');
          if (!readyCompleter.isCompleted) readyCompleter.complete();
        },
        onSpeechStateChanged: (isSpeaking) {
          _speechController.add(isSpeaking);
        },
        onSegmentDetected: (seg) {
          _segmentController.add(seg);
          _textUpdateController.add(
              TextUpdate(seg.index, seg.text, seg.elapsedSeconds));
        },
        onError: (msg) {
          _state = VadAsrState.uninitialized;
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

      await readyCompleter.future.timeout(
        const Duration(seconds: 60),
        onTimeout: () => throw TimeoutException('Web worker init timed out'),
      );
    } catch (e) {
      _state = VadAsrState.uninitialized;
      _logController.add('Init error: $e');
      rethrow;
    }
  }

  void acceptWaveform(Float32List samples) {
    if (_state != VadAsrState.initialized) return;
    _worker?.acceptWaveform(samples);
  }

  void reset() {
    _worker?.reset();
  }

  void cancel() {
    _worker?.reset();
  }

  void dispose() {
    _worker?.dispose();
    _worker = null;
    _state = VadAsrState.uninitialized;
    _logController.close();
    _segmentController.close();
    _textUpdateController.close();
    _speechController.close();
  }
}
