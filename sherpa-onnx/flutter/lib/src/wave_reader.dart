// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

class WaveData {
  WaveData({required this.samples, required this.sampleRate});

  /// normalized to [-1, 1]
  Float32List samples;
  int sampleRate;
}

WaveData readWave(String filename) {
  final Pointer<Utf8> str = filename.toNativeUtf8();
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
