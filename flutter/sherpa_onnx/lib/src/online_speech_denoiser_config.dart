// Copyright (c)  2026  Xiaomi Corporation
// Shared config/data classes for online speech denoising -- no FFI, works on all platforms.

import './offline_speech_denoiser_config.dart';

/// Streaming speech denoising.
///
/// Call [run] on consecutive chunks, then [flush] after the final chunk to
/// drain any buffered state.
class OnlineSpeechDenoiserConfig {
  const OnlineSpeechDenoiserConfig({
    this.model = const OfflineSpeechDenoiserModelConfig(),
  });

  factory OnlineSpeechDenoiserConfig.fromJson(Map<String, dynamic> json) {
    return OnlineSpeechDenoiserConfig(
      model: json['model'] != null
          ? OfflineSpeechDenoiserModelConfig.fromJson(
              json['model'] as Map<String, dynamic>,
            )
          : const OfflineSpeechDenoiserModelConfig(),
    );
  }

  @override
  String toString() {
    return 'OnlineSpeechDenoiserConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model.toJson(),
      };

  final OfflineSpeechDenoiserModelConfig model;
}
