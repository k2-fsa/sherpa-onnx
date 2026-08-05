// Copyright (c)  2024  Xiaomi Corporation
// Shared data classes for WAV reading -- no FFI, works on all platforms.
import 'dart:typed_data';

/// Audio samples loaded from a WAV file.
///
/// Samples are normalized to the range `[-1, 1]` and are stored as mono
/// `Float32List` PCM data.
class WaveData {
  WaveData({required this.samples, required this.sampleRate});

  /// normalized to [-1, 1]
  Float32List samples;
  int sampleRate;
}
