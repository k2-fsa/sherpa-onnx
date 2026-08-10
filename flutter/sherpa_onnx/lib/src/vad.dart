// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';
import './vad_config.dart';

export './vad_config.dart';

/// Circular sample buffer used by VAD-related pipelines.
class CircularBuffer {
  CircularBuffer.fromPtr({required this.ptr});

  CircularBuffer._({required this.ptr});

  /// The user has to invoke CircularBuffer.free() on the returned instance
  /// to avoid memory leak.
  factory CircularBuffer({required int capacity}) {
    assert(capacity > 0, 'capacity is $capacity');

    if (SherpaOnnxBindings.createCircularBuffer == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final p =
        SherpaOnnxBindings.createCircularBuffer?.call(capacity) ?? nullptr;

    if (p == nullptr) {
      throw Exception(
          "Failed to create circular buffer. Please check your config");
    }

    return CircularBuffer._(ptr: p);
  }

  /// Release the native buffer.
  void free() {
    if (SherpaOnnxBindings.destroyCircularBuffer == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroyCircularBuffer?.call(ptr);
    ptr = nullptr;
  }

  /// Append samples to the tail of the buffer.
  void push(Float32List data) {
    if (SherpaOnnxBindings.circularBufferPush == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }

    final n = data.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, data);

    SherpaOnnxBindings.circularBufferPush?.call(ptr, p, n);

    calloc.free(p);
  }

  /// Copy [n] samples starting at [startIndex].
  Float32List get({required int startIndex, required int n}) {
    if (SherpaOnnxBindings.circularBufferGet == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return Float32List(0);
    }

    final Pointer<Float> p =
        SherpaOnnxBindings.circularBufferGet?.call(ptr, startIndex, n) ??
            nullptr;

    if (p == nullptr) {
      return Float32List(0);
    }

    final pList = p.asTypedList(n);
    final Float32List ans = Float32List.fromList(pList);

    SherpaOnnxBindings.circularBufferFree?.call(p);

    return ans;
  }

  /// Drop [n] samples from the head of the buffer.
  void pop(int n) {
    if (SherpaOnnxBindings.circularBufferPop == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.circularBufferPop?.call(ptr, n);
  }

  /// Clear the buffer contents.
  void reset() {
    if (SherpaOnnxBindings.circularBufferReset == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.circularBufferReset?.call(ptr);
  }

  int get size {
    if (SherpaOnnxBindings.circularBufferSize == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.circularBufferSize?.call(ptr) ?? 0;
  }

  int get head {
    if (SherpaOnnxBindings.circularBufferHead == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.circularBufferHead?.call(ptr) ?? 0;
  }

  Pointer<SherpaOnnxCircularBuffer> ptr;
}

/// Voice activity detector that emits [SpeechSegment] objects.
class VoiceActivityDetector {
  VoiceActivityDetector.fromPtr({required this.ptr, required this.config});

  VoiceActivityDetector._({required this.ptr, required this.config});

  factory VoiceActivityDetector(
      {required VadModelConfig config, required double bufferSizeInSeconds}) {
    if (SherpaOnnxBindings.createVoiceActivityDetector == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final c = calloc<SherpaOnnxVadModelConfig>();

    final sileroVadModelPtr = config.sileroVad.model.toNativeUtf8();
    c.ref.sileroVad.model = sileroVadModelPtr;

    c.ref.sileroVad.threshold = config.sileroVad.threshold;
    c.ref.sileroVad.minSilenceDuration = config.sileroVad.minSilenceDuration;
    c.ref.sileroVad.minSpeechDuration = config.sileroVad.minSpeechDuration;
    c.ref.sileroVad.windowSize = config.sileroVad.windowSize;
    c.ref.sileroVad.maxSpeechDuration = config.sileroVad.maxSpeechDuration;

    final tenVadModelPtr = config.tenVad.model.toNativeUtf8();
    c.ref.tenVad.model = tenVadModelPtr;

    c.ref.tenVad.threshold = config.tenVad.threshold;
    c.ref.tenVad.minSilenceDuration = config.tenVad.minSilenceDuration;
    c.ref.tenVad.minSpeechDuration = config.tenVad.minSpeechDuration;
    c.ref.tenVad.windowSize = config.tenVad.windowSize;
    c.ref.tenVad.maxSpeechDuration = config.tenVad.maxSpeechDuration;

    c.ref.sampleRate = config.sampleRate;
    c.ref.numThreads = config.numThreads;

    final providerPtr = config.provider.toNativeUtf8();
    c.ref.provider = providerPtr;

    c.ref.debug = config.debug ? 1 : 0;

    final ptr = SherpaOnnxBindings.createVoiceActivityDetector
            ?.call(c, bufferSizeInSeconds) ??
        nullptr;

    calloc.free(providerPtr);
    calloc.free(tenVadModelPtr);
    calloc.free(sileroVadModelPtr);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception("Failed to create vad. Please check your config");
    }

    return VoiceActivityDetector._(ptr: ptr, config: config);
  }

  void free() {
    if (SherpaOnnxBindings.destroyVoiceActivityDetector == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroyVoiceActivityDetector?.call(ptr);
    ptr = nullptr;
  }

  void acceptWaveform(Float32List samples) {
    if (SherpaOnnxBindings.voiceActivityDetectorAcceptWaveform == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }

    final n = samples.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, samples);

    SherpaOnnxBindings.voiceActivityDetectorAcceptWaveform?.call(ptr, p, n);

    calloc.free(p);
  }

  bool isEmpty() {
    if (SherpaOnnxBindings.voiceActivityDetectorEmpty == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return true;
    }

    final int empty =
        SherpaOnnxBindings.voiceActivityDetectorEmpty?.call(ptr) ?? 0;

    return empty == 1;
  }

  bool isDetected() {
    if (SherpaOnnxBindings.voiceActivityDetectorDetected == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return false;
    }

    final int detected =
        SherpaOnnxBindings.voiceActivityDetectorDetected?.call(ptr) ?? 0;

    return detected == 1;
  }

  void pop() {
    if (SherpaOnnxBindings.voiceActivityDetectorPop == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.voiceActivityDetectorPop?.call(ptr);
  }

  void clear() {
    if (SherpaOnnxBindings.voiceActivityDetectorClear == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.voiceActivityDetectorClear?.call(ptr);
  }

  SpeechSegment front() {
    if (SherpaOnnxBindings.voiceActivityDetectorFront == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return SpeechSegment(samples: Float32List(0), start: 0);
    }

    final Pointer<SherpaOnnxSpeechSegment> segment =
        SherpaOnnxBindings.voiceActivityDetectorFront?.call(ptr) ?? nullptr;
    if (segment == nullptr) {
      return SpeechSegment(samples: Float32List(0), start: 0);
    }

    final sampleList = segment.ref.samples.asTypedList(segment.ref.n);
    final start = segment.ref.start;

    final samples = Float32List.fromList(sampleList);

    SherpaOnnxBindings.destroySpeechSegment?.call(segment);

    return SpeechSegment(samples: samples, start: start);
  }

  void reset() {
    if (SherpaOnnxBindings.voiceActivityDetectorReset == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.voiceActivityDetectorReset?.call(ptr);
  }

  void flush() {
    if (SherpaOnnxBindings.voiceActivityDetectorFlush == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.voiceActivityDetectorFlush?.call(ptr);
  }

  Pointer<SherpaOnnxVoiceActivityDetector> ptr;
  final VadModelConfig config;
}
