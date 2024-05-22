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
      this.windowSize = 512});

  @override
  String toString() {
    return 'SileroVadModelConfig(model: $model, threshold: $threshold, minSilenceDuration: $minSilenceDuration, minSpeechDuration: $minSpeechDuration, windowSize: $windowSize)';
  }

  final String model;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final int windowSize;
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

class CircularBuffer {
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

  Pointer<SherpaOnnxCircularBuffer> ptr;
}
