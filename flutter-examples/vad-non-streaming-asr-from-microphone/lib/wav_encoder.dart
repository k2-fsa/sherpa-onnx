// Copyright (c)  2026  Xiaomi Corporation
// Shared WAV encoding/decoding utility — works on all platforms (including web).
import 'dart:typed_data';

/// Result of decoding a WAV file.
class WavData {
  final Float32List samples;
  final int sampleRate;
  const WavData({required this.samples, required this.sampleRate});
}

/// Decode WAV bytes to mono float samples + sample rate.
/// Supports 16-bit PCM and 32-bit float PCM.
/// Returns null if the format is unsupported or the file is invalid.
WavData? decodeWav(Uint8List bytes) {
  if (bytes.length < 44) return null;
  final bd = bytes.buffer.asByteData(bytes.offsetInBytes, bytes.lengthInBytes);

  // Check RIFF/WAVE header.
  if (bytes[0] != 0x52 || bytes[1] != 0x49 || bytes[2] != 0x46 || bytes[3] != 0x46) {
    return null; // Not RIFF
  }
  if (bytes[8] != 0x57 || bytes[9] != 0x41 || bytes[10] != 0x56 || bytes[11] != 0x45) {
    return null; // Not WAVE
  }

  // Find "fmt " chunk.
  int offset = 12;
  int audioFormat = 0;
  int numChannels = 0;
  int sampleRate = 0;
  int bitsPerSample = 0;
  bool foundFmt = false;

  while (offset + 8 <= bytes.length) {
    final chunkId = String.fromCharCodes(bytes.sublist(offset, offset + 4));
    final chunkSize = bd.getUint32(offset + 4, Endian.little);
    if (chunkId == 'fmt ') {
      audioFormat = bd.getUint16(offset + 8, Endian.little);
      numChannels = bd.getUint16(offset + 10, Endian.little);
      sampleRate = bd.getUint32(offset + 12, Endian.little);
      bitsPerSample = bd.getUint16(offset + 22, Endian.little);
      foundFmt = true;
      offset += 8 + chunkSize;
      break;
    }
    offset += 8 + chunkSize;
  }
  if (!foundFmt) return null;

  // Find "data" chunk.
  offset = 12;
  Uint8List? dataBytes;
  while (offset + 8 <= bytes.length) {
    final chunkId = String.fromCharCodes(bytes.sublist(offset, offset + 4));
    final chunkSize = bd.getUint32(offset + 4, Endian.little);
    if (chunkId == 'data') {
      dataBytes = Uint8List.view(bytes.buffer, bytes.offsetInBytes + offset + 8, chunkSize);
      break;
    }
    offset += 8 + chunkSize;
  }
  if (dataBytes == null) return null;

  // Decode to mono float samples.
  if (audioFormat == 1 && bitsPerSample == 16) {
    // 16-bit PCM
    final numSamples = dataBytes.length ~/ (2 * numChannels);
    final samples = Float32List(numSamples);
    final dbd = dataBytes.buffer.asByteData(dataBytes.offsetInBytes, dataBytes.lengthInBytes);
    for (int i = 0; i < numSamples; i++) {
      // Mix to mono if stereo: average channels.
      double sum = 0;
      for (int ch = 0; ch < numChannels; ch++) {
        sum += dbd.getInt16((i * numChannels + ch) * 2, Endian.little) / 32768.0;
      }
      samples[i] = sum / numChannels;
    }
    return WavData(samples: samples, sampleRate: sampleRate);
  } else if (audioFormat == 3 && bitsPerSample == 32) {
    // 32-bit float PCM
    final numSamples = dataBytes.length ~/ (4 * numChannels);
    final samples = Float32List(numSamples);
    final dbd = dataBytes.buffer.asByteData(dataBytes.offsetInBytes, dataBytes.lengthInBytes);
    for (int i = 0; i < numSamples; i++) {
      double sum = 0;
      for (int ch = 0; ch < numChannels; ch++) {
        sum += dbd.getFloat32((i * numChannels + ch) * 4, Endian.little);
      }
      samples[i] = sum / numChannels;
    }
    return WavData(samples: samples, sampleRate: sampleRate);
  }

  return null; // Unsupported format
}

/// Encode Float32List PCM samples to WAV bytes (16-bit mono).
Uint8List encodeWav(Float32List samples, int sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
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
  bd.setUint32(16, 16, Endian.little);
  bd.setUint16(20, 1, Endian.little); // PCM
  bd.setUint16(22, numChannels, Endian.little);
  bd.setUint32(24, sampleRate, Endian.little);
  bd.setUint32(28, byteRate, Endian.little);
  bd.setUint16(32, blockAlign, Endian.little);
  bd.setUint16(34, bitsPerSample, Endian.little);

  // data chunk
  buffer.setRange(36, 40, [0x64, 0x61, 0x74, 0x61]); // "data"
  bd.setUint32(40, dataSize, Endian.little);

  for (int i = 0; i < samples.length; i++) {
    final s = (samples[i] * 32767).clamp(-32768, 32767).toInt();
    bd.setInt16(44 + i * 2, s, Endian.little);
  }

  return buffer;
}
