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
    this.attenuationLimitDb = 0.0,
  });

  factory OfflineSpeechDenoiserDpdfNetModelConfig.fromJson(
      Map<String, dynamic> json) {
    return OfflineSpeechDenoiserDpdfNetModelConfig(
      model: json['model'] as String? ?? '',
      attenuationLimitDb:
          (json['attenuation_limit_db'] as num?)?.toDouble() ?? 0.0,
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserDpdfNetModelConfig(model: $model, '
        'attenuationLimitDb: $attenuationLimitDb)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'attenuation_limit_db': attenuationLimitDb,
      };

  final String model;
  final double attenuationLimitDb;
}

/// VoxLock model config for offline speech denoising.
///
/// VoxLock keeps the voice of one speaker (the caller) and removes the other
/// voices. The model is from https://github.com/arbazsiddiqui/voxlock
class OfflineSpeechDenoiserVoxLockModelConfig {
  const OfflineSpeechDenoiserVoxLockModelConfig({
    this.enrollModel = '',
    this.chunkModel = '',
    this.voiceSample = '',
  });

  factory OfflineSpeechDenoiserVoxLockModelConfig.fromJson(
      Map<String, dynamic> json) {
    return OfflineSpeechDenoiserVoxLockModelConfig(
      enrollModel: json['enroll_model'] as String? ?? '',
      chunkModel: json['chunk_model'] as String? ?? '',
      voiceSample: json['voice_sample'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserVoxLockModelConfig(enrollModel: $enrollModel, '
        'chunkModel: $chunkModel, voiceSample: $voiceSample)';
  }

  Map<String, dynamic> toJson() => {
        'enroll_model': enrollModel,
        'chunk_model': chunkModel,
        'voice_sample': voiceSample,
      };

  /// Path to the VoxLock enroll model, e.g., enroll.onnx. It runs once on a
  /// voice sample of the speaker to keep.
  final String enrollModel;

  /// Path to the VoxLock chunk model, e.g., chunk.onnx or chunk_int8.onnx. It
  /// runs on each chunk of the audio.
  final String chunkModel;

  /// Optional. Path to a wave file with at least 3 seconds of the caller's
  /// voice. VoxLock keeps this voice and removes the others. From a longer
  /// file, it uses the loudest 3 seconds within the first 30 seconds. A file
  /// that is shorter or cannot be read is ignored, as if it were empty.
  ///
  /// If empty or ignored, the voice sample is taken from the input audio: the
  /// offline denoiser uses the loudest 3 seconds in the first 15 seconds of
  /// the input, and the online denoiser passes the audio through unchanged
  /// until 3 seconds of speech have been heard, then uses that speech.
  final String voiceSample;
}

/// Aggregate model configuration for [OfflineSpeechDenoiser].
///
/// Configure exactly one of [gtcrn], [dpdfnet] or [voxlock] for typical use.
class OfflineSpeechDenoiserModelConfig {
  const OfflineSpeechDenoiserModelConfig({
    this.gtcrn = const OfflineSpeechDenoiserGtcrnModelConfig(),
    this.dpdfnet = const OfflineSpeechDenoiserDpdfNetModelConfig(),
    this.voxlock = const OfflineSpeechDenoiserVoxLockModelConfig(),
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
      voxlock: json['voxlock'] != null
          ? OfflineSpeechDenoiserVoxLockModelConfig.fromJson(
              json['voxlock'] as Map<String, dynamic>)
          : const OfflineSpeechDenoiserVoxLockModelConfig(),
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserModelConfig(gtcrn: $gtcrn, dpdfnet: $dpdfnet, '
        'voxlock: $voxlock, numThreads: $numThreads, debug: $debug, '
        'provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'gtcrn': gtcrn.toJson(),
        'dpdfnet': dpdfnet.toJson(),
        'voxlock': voxlock.toJson(),
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final OfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  final OfflineSpeechDenoiserDpdfNetModelConfig dpdfnet;
  final OfflineSpeechDenoiserVoxLockModelConfig voxlock;
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
