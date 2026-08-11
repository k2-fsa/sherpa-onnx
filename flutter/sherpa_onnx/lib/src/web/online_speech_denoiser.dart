// Copyright (c)  2026  Xiaomi Corporation
// Web stub for online speech denoiser -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import 'dart:typed_data';

import '../offline_speech_denoiser_config.dart';
import '../online_speech_denoiser_config.dart';

export '../online_speech_denoiser_config.dart';

/// Streaming speech denoiser.
class OnlineSpeechDenoiser {
  OnlineSpeechDenoiser.fromPtr({required this.ptr, required this.config});

  factory OnlineSpeechDenoiser(OnlineSpeechDenoiserConfig config) {
    throw UnsupportedError(
        'OnlineSpeechDenoiser is not yet supported on web');
  }

  void free() {}
  DenoisedAudio run({required Float32List samples, required int sampleRate}) =>
      DenoisedAudio(samples: Float32List(0), sampleRate: 0);
  DenoisedAudio flush() =>
      DenoisedAudio(samples: Float32List(0), sampleRate: 0);
  void reset() {}

  int get sampleRate => 0;
  int get frameShiftInSamples => 0;

  dynamic ptr;
  OnlineSpeechDenoiserConfig config;
}
