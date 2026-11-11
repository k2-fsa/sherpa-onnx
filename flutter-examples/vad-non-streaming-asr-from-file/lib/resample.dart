// Copyright (c)  2026  Xiaomi Corporation
// Simple linear interpolation resampler to 16kHz.
import 'dart:typed_data';

/// Resample [samples] from [srcRate] to 16kHz using linear interpolation.
/// Returns [samples] unchanged if already at 16kHz.
Float32List resampleTo16k(Float32List samples, int srcRate) {
  if (srcRate == 16000) return samples;

  final ratio = 16000 / srcRate;
  final outLen = (samples.length * ratio).round();
  final out = Float32List(outLen);
  for (int i = 0; i < outLen; i++) {
    final srcPos = i / ratio;
    final idx = srcPos.floor();
    final frac = srcPos - idx;
    if (idx + 1 < samples.length) {
      out[i] = samples[idx] * (1 - frac) + samples[idx + 1] * frac;
    } else if (idx < samples.length) {
      out[i] = samples[idx];
    }
  }
  return out;
}
