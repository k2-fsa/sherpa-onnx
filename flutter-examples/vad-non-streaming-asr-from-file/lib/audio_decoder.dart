// Copyright (c)  2026  Xiaomi Corporation
// Audio decoder — uses FFmpeg on native, Web Audio API on web.
import 'dart:typed_data';

/// Result of decoding an audio file.
class DecodedAudio {
  final Float32List samples;
  final int sampleRate;
  final double duration;
  const DecodedAudio({
    required this.samples,
    required this.sampleRate,
    required this.duration,
  });
}

/// Decode an audio file to 16kHz mono Float32 PCM samples.
/// Returns null if decoding fails.
Future<DecodedAudio?> decodeAudioFile(String filePath) async {
  // This is the native implementation.
  // On web, a separate implementation is needed.
  throw UnsupportedError('decodeAudioFile is not supported on this platform');
}
