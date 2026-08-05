// Copyright (c)  2026  Xiaomi Corporation
// Web-specific utilities.
import 'dart:js_interop';
import 'dart:typed_data';

/// Encode Float32List samples to WAV bytes.
Uint8List encodeWav(Float32List samples, int sampleRate) {
  final numChannels = 1;
  final bitsPerSample = 16;
  final byteRate = sampleRate * numChannels * bitsPerSample ~/ 8;
  final blockAlign = numChannels * bitsPerSample ~/ 8;
  final dataSize = samples.length * 2;
  final totalSize = 44 + dataSize;

  final buffer = Uint8List(totalSize);
  final bd = buffer.buffer.asByteData();

  // RIFF header
  buffer.setRange(0, 4, [0x52, 0x49, 0x46, 0x46]); // "RIFF"
  bd.setUint32(4, totalSize - 8, Endian.little);
  buffer.setRange(8, 12, [0x57, 0x41, 0x56, 0x45]); // "WAVE"

  // fmt chunk
  buffer.setRange(12, 16, [0x66, 0x6d, 0x74, 0x20]); // "fmt "
  bd.setUint32(16, 16, Endian.little); // chunk size
  bd.setUint16(20, 1, Endian.little); // PCM
  bd.setUint16(22, numChannels, Endian.little);
  bd.setUint32(24, sampleRate, Endian.little);
  bd.setUint32(28, byteRate, Endian.little);
  bd.setUint16(32, blockAlign, Endian.little);
  bd.setUint16(34, bitsPerSample, Endian.little);

  // data chunk
  buffer.setRange(36, 40, [0x64, 0x61, 0x74, 0x61]); // "data"
  bd.setUint32(40, dataSize, Endian.little);

  // Convert float samples to 16-bit PCM.
  for (int i = 0; i < samples.length; i++) {
    final s = (samples[i] * 32767).clamp(-32768, 32767).toInt();
    bd.setInt16(44 + i * 2, s, Endian.little);
  }

  return buffer;
}

/// Generate a filename (on web, this is just a hint for display).
Future<String> generateWaveFilename([String suffix = '']) async {
  DateTime now = DateTime.now();
  return '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}-${now.hour.toString().padLeft(2, '0')}-${now.minute.toString().padLeft(2, '0')}-${now.second.toString().padLeft(2, '0')}$suffix.wav';
}
