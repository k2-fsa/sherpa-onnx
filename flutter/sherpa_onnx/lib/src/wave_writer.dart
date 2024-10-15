// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

bool writeWave(
    {required String filename,
    required Float32List samples,
    required int sampleRate}) {
  final Pointer<Utf8> filenamePtr = filename.toNativeUtf8();

  final n = samples.length;
  final Pointer<Float> p = calloc<Float>(n);

  final pList = p.asTypedList(n);
  pList.setAll(0, samples);

  int ok =
      SherpaOnnxBindings.writeWave?.call(p, n, sampleRate, filenamePtr) ?? 0;

  calloc.free(p);
  calloc.free(filenamePtr);

  return ok == 1;
}
