// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

class SileroVadModelConfig {
  const SileroVadModelConfig(
      {this.model = '',
      this.threshold = 0.5,
      this.minSilenceDuration = 0.5,
      this.minSpeechDuration = 0.25,
      this.windowSize = 512,
      this.maxSpeechDuration = 5.0});

  @override
  String toString() {
    return 'SileroVadModelConfig(model: $model, threshold: $threshold, minSilenceDuration: $minSilenceDuration, minSpeechDuration: $minSpeechDuration, windowSize: $windowSize, maxSpeechDuration: $maxSpeechDuration)';
  }

  final String model;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final int windowSize;
  final double maxSpeechDuration;
}

class VadModelConfig {
  VadModelConfig(
      {this.sileroVad = const SileroVadModelConfig(),
      this.sampleRate = 16000,
      this.numThreads = 1,
      this.provider = 'cpu',
      this.debug = true});

  @override
  String toString() {
    return 'VadModelConfig(sileroVad: $sileroVad, sampleRate: $sampleRate, numThreads: $numThreads, provider: $provider, debug: $debug)';
  }

  final SileroVadModelConfig sileroVad;
  final int sampleRate;
  final int numThreads;
  final String provider;
  final bool debug;
}

class SpeechSegment {
  SpeechSegment({required this.samples, required this.start});
  final Float32List samples;
  final int start;
}

class CircularBuffer {
  CircularBuffer.fromPtr({required this.ptr});

  CircularBuffer._({required this.ptr});

  /// The user has to invoke CircularBuffer.free() on the returned instance
  /// to avoid memory leak.
  factory CircularBuffer({required int capacity}) {
    assert(capacity > 0, 'capacity is $capacity');
    final p =
        SherpaOnnxBindings.createCircularBuffer?.call(capacity) ?? nullptr;

    return CircularBuffer._(ptr: p);
  }

  void free() {
    SherpaOnnxBindings.destroyCircularBuffer?.call(ptr);
    ptr = nullptr;
  }

  void push(Float32List data) {
    final n = data.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, data);

    SherpaOnnxBindings.circularBufferPush?.call(ptr, p, n);

    calloc.free(p);
  }

  Float32List get({required int startIndex, required int n}) {
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

  void pop(int n) {
    SherpaOnnxBindings.circularBufferPop?.call(ptr, n);
  }

  void reset() {
    SherpaOnnxBindings.circularBufferReset?.call(ptr);
  }

  int get size => SherpaOnnxBindings.circularBufferSize?.call(ptr) ?? 0;
  int get head => SherpaOnnxBindings.circularBufferHead?.call(ptr) ?? 0;

  Pointer<SherpaOnnxCircularBuffer> ptr;
}

class VoiceActivityDetector {
  VoiceActivityDetector.fromPtr({required this.ptr, required this.config});

  VoiceActivityDetector._({required this.ptr, required this.config});

  // The user has to invoke VoiceActivityDetector.free() to avoid memory leak.
  factory VoiceActivityDetector(
      {required VadModelConfig config, required double bufferSizeInSeconds}) {
    final c = calloc<SherpaOnnxVadModelConfig>();

    final modelPtr = config.sileroVad.model.toNativeUtf8();
    c.ref.sileroVad.model = modelPtr;

    c.ref.sileroVad.threshold = config.sileroVad.threshold;
    c.ref.sileroVad.minSilenceDuration = config.sileroVad.minSilenceDuration;
    c.ref.sileroVad.minSpeechDuration = config.sileroVad.minSpeechDuration;
    c.ref.sileroVad.windowSize = config.sileroVad.windowSize;
    c.ref.sileroVad.maxSpeechDuration = config.sileroVad.maxSpeechDuration;

    c.ref.sampleRate = config.sampleRate;
    c.ref.numThreads = config.numThreads;

    final providerPtr = config.provider.toNativeUtf8();
    c.ref.provider = providerPtr;

    c.ref.debug = config.debug ? 1 : 0;

    final ptr = SherpaOnnxBindings.createVoiceActivityDetector
            ?.call(c, bufferSizeInSeconds) ??
        nullptr;

    calloc.free(providerPtr);
    calloc.free(modelPtr);
    calloc.free(c);

    return VoiceActivityDetector._(ptr: ptr, config: config);
  }

  void free() {
    SherpaOnnxBindings.destroyVoiceActivityDetector?.call(ptr);
    ptr = nullptr;
  }

  void acceptWaveform(Float32List samples) {
    final n = samples.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, samples);

    SherpaOnnxBindings.voiceActivityDetectorAcceptWaveform?.call(ptr, p, n);

    calloc.free(p);
  }

  bool isEmpty() {
    final int empty =
        SherpaOnnxBindings.voiceActivityDetectorEmpty?.call(ptr) ?? 0;

    return empty == 1;
  }

  bool isDetected() {
    final int detected =
        SherpaOnnxBindings.voiceActivityDetectorDetected?.call(ptr) ?? 0;

    return detected == 1;
  }

  void pop() {
    SherpaOnnxBindings.voiceActivityDetectorPop?.call(ptr);
  }

  void clear() {
    SherpaOnnxBindings.voiceActivityDetectorClear?.call(ptr);
  }

  SpeechSegment front() {
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
    SherpaOnnxBindings.voiceActivityDetectorReset?.call(ptr);
  }

  void flush() {
    SherpaOnnxBindings.voiceActivityDetectorFlush?.call(ptr);
  }

  Pointer<SherpaOnnxVoiceActivityDetector> ptr;
  final VadModelConfig config;
}
