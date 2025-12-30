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

  factory SileroVadModelConfig.fromJson(Map<String, dynamic> json) {
    return SileroVadModelConfig(
      model: json['model'] as String? ?? '',
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
      minSilenceDuration:
          (json['minSilenceDuration'] as num?)?.toDouble() ?? 0.5,
      minSpeechDuration:
          (json['minSpeechDuration'] as num?)?.toDouble() ?? 0.25,
      windowSize: json['windowSize'] as int? ?? 512,
      maxSpeechDuration: (json['maxSpeechDuration'] as num?)?.toDouble() ?? 5.0,
    );
  }

  @override
  String toString() {
    return 'SileroVadModelConfig(model: $model, threshold: $threshold, minSilenceDuration: $minSilenceDuration, minSpeechDuration: $minSpeechDuration, windowSize: $windowSize, maxSpeechDuration: $maxSpeechDuration)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'threshold': threshold,
        'minSilenceDuration': minSilenceDuration,
        'minSpeechDuration': minSpeechDuration,
        'windowSize': windowSize,
        'maxSpeechDuration': maxSpeechDuration,
      };

  final String model;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final int windowSize;
  final double maxSpeechDuration;
}

class TenVadModelConfig {
  const TenVadModelConfig(
      {this.model = '',
      this.threshold = 0.5,
      this.minSilenceDuration = 0.5,
      this.minSpeechDuration = 0.25,
      this.windowSize = 256,
      this.maxSpeechDuration = 5.0});

  factory TenVadModelConfig.fromJson(Map<String, dynamic> json) {
    return TenVadModelConfig(
      model: json['model'] as String? ?? '',
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
      minSilenceDuration:
          (json['minSilenceDuration'] as num?)?.toDouble() ?? 0.5,
      minSpeechDuration:
          (json['minSpeechDuration'] as num?)?.toDouble() ?? 0.25,
      windowSize: json['windowSize'] as int? ?? 256,
      maxSpeechDuration: (json['maxSpeechDuration'] as num?)?.toDouble() ?? 5.0,
    );
  }

  @override
  String toString() {
    return 'TenVadModelConfig(model: $model, threshold: $threshold, minSilenceDuration: $minSilenceDuration, minSpeechDuration: $minSpeechDuration, windowSize: $windowSize, maxSpeechDuration: $maxSpeechDuration)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'threshold': threshold,
        'minSilenceDuration': minSilenceDuration,
        'minSpeechDuration': minSpeechDuration,
        'windowSize': windowSize,
        'maxSpeechDuration': maxSpeechDuration,
      };

  final String model;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final int windowSize;
  final double maxSpeechDuration;
}

class VadModelConfig {
  VadModelConfig({
    this.sileroVad = const SileroVadModelConfig(),
    this.sampleRate = 16000,
    this.numThreads = 1,
    this.provider = 'cpu',
    this.debug = true,
    this.tenVad = const TenVadModelConfig(),
  });

  final SileroVadModelConfig sileroVad;
  final TenVadModelConfig tenVad;
  final int sampleRate;
  final int numThreads;
  final String provider;
  final bool debug;

  factory VadModelConfig.fromJson(Map<String, dynamic> json) {
    return VadModelConfig(
      sileroVad: SileroVadModelConfig.fromJson(
          json['sileroVad'] as Map<String, dynamic>? ?? const {}),
      tenVad: TenVadModelConfig.fromJson(
          json['tenVad'] as Map<String, dynamic>? ?? const {}),
      sampleRate: json['sampleRate'] as int? ?? 16000,
      numThreads: json['numThreads'] as int? ?? 1,
      provider: json['provider'] as String? ?? 'cpu',
      debug: json['debug'] as bool? ?? true,
    );
  }

  Map<String, dynamic> toJson() => {
        'sileroVad': sileroVad.toJson(),
        'tenVad': tenVad.toJson(),
        'sampleRate': sampleRate,
        'numThreads': numThreads,
        'provider': provider,
        'debug': debug,
      };

  @override
  String toString() {
    return 'VadModelConfig(sileroVad: $sileroVad, tenVad: $tenVad, sampleRate: $sampleRate, numThreads: $numThreads, provider: $provider, debug: $debug)';
  }
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
