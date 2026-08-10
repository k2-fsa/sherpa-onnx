// Copyright (c)  2026  Xiaomi Corporation
// Web stub for offline speech denoiser -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import 'dart:typed_data';

import '../offline_speech_denoiser_config.dart';

export '../offline_speech_denoiser_config.dart';

/// Offline speech denoiser.
class OfflineSpeechDenoiser {
  OfflineSpeechDenoiser.fromPtr({required this.ptr, required this.config});
  OfflineSpeechDenoiser._({required this.ptr, required this.config});

  factory OfflineSpeechDenoiser(OfflineSpeechDenoiserConfig config) {
    throw UnsupportedError(
        'OfflineSpeechDenoiser is not yet supported on web');
  }

  void free() {}
  DenoisedAudio run({required Float32List samples, required int sampleRate}) =>
      DenoisedAudio(samples: Float32List(0), sampleRate: 0);

  int get sampleRate => 0;

  dynamic ptr;
  OfflineSpeechDenoiserConfig config;
}
