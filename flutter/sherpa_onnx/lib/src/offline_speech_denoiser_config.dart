// Copyright (c)  2025  Xiaomi Corporation
// Shared config/data classes for offline speech denoising -- no FFI, works on all platforms.
import 'dart:typed_data';

/// Offline speech denoising.
///
/// Supported model families include GTCRN and DPDFNet. See the examples under
/// `dart-api-examples/speech-enhancement-gtcrn/` and
/// `dart-api-examples/speech-enhancement-dpdfnet/`.
class OfflineSpeechDenoiserGtcrnModelConfig {
  const OfflineSpeechDenoiserGtcrnModelConfig({
    this.model = '',
  });

  factory OfflineSpeechDenoiserGtcrnModelConfig.fromJson(
      Map<String, dynamic> json) {
    return OfflineSpeechDenoiserGtcrnModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserGtcrnModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

/// DPDFNet model path for offline speech denoising.
class OfflineSpeechDenoiserDpdfNetModelConfig {
  const OfflineSpeechDenoiserDpdfNetModelConfig({
    this.model = '',
  });

  factory OfflineSpeechDenoiserDpdfNetModelConfig.fromJson(
      Map<String, dynamic> json) {
    return OfflineSpeechDenoiserDpdfNetModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserDpdfNetModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

/// Aggregate model configuration for [OfflineSpeechDenoiser].
///
/// Configure either [gtcrn] or [dpdfnet] for typical use.
class OfflineSpeechDenoiserModelConfig {
  const OfflineSpeechDenoiserModelConfig({
    this.gtcrn = const OfflineSpeechDenoiserGtcrnModelConfig(),
    this.dpdfnet = const OfflineSpeechDenoiserDpdfNetModelConfig(),
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
  });

  factory OfflineSpeechDenoiserModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineSpeechDenoiserModelConfig(
      gtcrn: json['gtcrn'] != null
          ? OfflineSpeechDenoiserGtcrnModelConfig.fromJson(
              json['gtcrn'] as Map<String, dynamic>)
          : const OfflineSpeechDenoiserGtcrnModelConfig(),
      dpdfnet: json['dpdfnet'] != null
          ? OfflineSpeechDenoiserDpdfNetModelConfig.fromJson(
              json['dpdfnet'] as Map<String, dynamic>)
          : const OfflineSpeechDenoiserDpdfNetModelConfig(),
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserModelConfig(gtcrn: $gtcrn, dpdfnet: $dpdfnet, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'gtcrn': gtcrn.toJson(),
        'dpdfnet': dpdfnet.toJson(),
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final OfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  final OfflineSpeechDenoiserDpdfNetModelConfig dpdfnet;
  final int numThreads;
  final bool debug;
  final String provider;
}

/// Top-level configuration for [OfflineSpeechDenoiser].
class OfflineSpeechDenoiserConfig {
  const OfflineSpeechDenoiserConfig({
    this.model = const OfflineSpeechDenoiserModelConfig(),
  });

  factory OfflineSpeechDenoiserConfig.fromJson(Map<String, dynamic> json) {
    return OfflineSpeechDenoiserConfig(
      model: json['model'] != null
          ? OfflineSpeechDenoiserModelConfig.fromJson(
              json['model'] as Map<String, dynamic>)
          : const OfflineSpeechDenoiserModelConfig(),
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model.toJson(),
      };

  final OfflineSpeechDenoiserModelConfig model;
}

/// Audio returned by offline or online speech denoisers.
class DenoisedAudio {
  DenoisedAudio({
    required this.samples,
    required this.sampleRate,
  });

  final Float32List samples;
  final int sampleRate;
}
