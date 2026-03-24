// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

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

/// Read a WAV file from disk.
///
/// Returns an empty [WaveData] object if the file cannot be read or decoded.
WaveData readWave(String filename) {
  final Pointer<Utf8> str = filename.toNativeUtf8();

  if (SherpaOnnxBindings.readWave == null) {
    throw Exception("Please initialize sherpa-onnx first");
  }

  Pointer<SherpaOnnxWave> wave =
      SherpaOnnxBindings.readWave?.call(str) ?? nullptr;
  calloc.free(str);

  if (wave == nullptr) {
    return WaveData(samples: Float32List(0), sampleRate: 0);
  }

  final samples = wave.ref.samples.asTypedList(wave.ref.numSamples);

  final newSamples = Float32List.fromList(samples);
  int sampleRate = wave.ref.sampleRate;
  SherpaOnnxBindings.freeWave?.call(wave);

  return WaveData(samples: newSamples, sampleRate: sampleRate);
}
