// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for VAD — no FFI, works on all platforms.
import 'dart:typed_data';

/// Silero VAD model configuration.
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

/// Ten VAD model configuration.
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

/// Top-level VAD model configuration.
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

/// One detected speech segment emitted by VoiceActivityDetector.
class SpeechSegment {
  SpeechSegment({required this.samples, required this.start});
  final Float32List samples;
  final int start;
}
