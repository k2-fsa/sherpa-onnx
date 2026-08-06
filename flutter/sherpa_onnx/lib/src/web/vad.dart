// Copyright (c)  2026  Xiaomi Corporation
// Web implementation of VAD using dart:js_interop.
import 'dart:typed_data';

import '../vad_config.dart';

export '../vad_config.dart';

/// Circular sample buffer used by VAD-related pipelines.
class CircularBuffer {
  CircularBuffer.fromPtr({required this.ptr});
  CircularBuffer._({required this.ptr});

  factory CircularBuffer({required int capacity}) {
    throw UnsupportedError('CircularBuffer is not yet supported on web');
  }

  void free() {}
  void push(Float32List data) {}
  Float32List get({required int startIndex, required int n}) =>
      Float32List(0);
  void pop(int n) {}
  void reset() {}
  int get size => 0;
  int get head => 0;
  dynamic ptr;
}

/// Voice activity detector that emits [SpeechSegment] objects.
class VoiceActivityDetector {
  VoiceActivityDetector.fromPtr({required this.ptr, required this.config});
  VoiceActivityDetector._({required this.ptr, required this.config});

  factory VoiceActivityDetector(
      {required VadModelConfig config, required double bufferSizeInSeconds}) {
    throw UnsupportedError(
        'VoiceActivityDetector is not yet supported on web');
  }

  void free() {}
  void acceptWaveform(Float32List samples) {}
  bool isEmpty() => true;
  bool isDetected() => false;
  void pop() {}
  void clear() {}
  SpeechSegment front() => SpeechSegment(samples: Float32List(0), start: 0);
  void reset() {}
  void flush() {}

  dynamic ptr;
  VadModelConfig config;
}
