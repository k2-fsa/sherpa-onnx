// Copyright (c)  2026  Xiaomi Corporation
// Web stub for offline speaker diarization -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import 'dart:typed_data';

import '../offline_speaker_diarization_config.dart';

export '../offline_speaker_diarization_config.dart';

/// Offline speaker diarizer.
class OfflineSpeakerDiarization {
  OfflineSpeakerDiarization.fromPtr(
      {required this.ptr, required this.config, required this.sampleRate});
  OfflineSpeakerDiarization._(
      {required this.ptr, required this.config, required this.sampleRate});

  factory OfflineSpeakerDiarization(OfflineSpeakerDiarizationConfig config) {
    throw UnsupportedError(
        'OfflineSpeakerDiarization is not yet supported on web');
  }

  void free() {}
  List<OfflineSpeakerDiarizationSegment> process(
          {required Float32List samples}) =>
      <OfflineSpeakerDiarizationSegment>[];
  List<OfflineSpeakerDiarizationSegment> processWithCallback({
    required Float32List samples,
    required int Function(int numProcessedChunks, int numTotalChunks) callback,
  }) =>
      <OfflineSpeakerDiarizationSegment>[];

  dynamic ptr;
  OfflineSpeakerDiarizationConfig config;
  final int sampleRate;
}
