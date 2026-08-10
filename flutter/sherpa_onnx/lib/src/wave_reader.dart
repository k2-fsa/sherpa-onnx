// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';
import './wave_reader_config.dart';

export './wave_reader_config.dart';

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
