// Copyright (c)  2026  Xiaomi Corporation
// Web-specific utilities.
import './wav_encoder.dart';
export './wav_encoder.dart' show encodeWav;

/// Generate a filename (on web, this is just a hint for display).
Future<String> generateWaveFilename([String suffix = '']) async {
  DateTime now = DateTime.now();
  return '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}-${now.hour.toString().padLeft(2, '0')}-${now.minute.toString().padLeft(2, '0')}-${now.second.toString().padLeft(2, '0')}$suffix.wav';
}
