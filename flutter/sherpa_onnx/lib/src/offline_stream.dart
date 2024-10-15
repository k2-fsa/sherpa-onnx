// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

class OfflineStream {
  /// The user has to call OfflineStream.free() to avoid memory leak.
  OfflineStream({required this.ptr});

  void free() {
    SherpaOnnxBindings.destroyOfflineStream?.call(ptr);
    ptr = nullptr;
  }

  /// If you have List<double> data, then you can use
  /// Float32List.fromList(data) to convert data to Float32List
  ///
  /// See
  ///  https://api.flutter.dev/flutter/dart-core/List-class.html
  /// and
  ///  https://api.flutter.dev/flutter/dart-typed_data/Float32List-class.html
  void acceptWaveform({required Float32List samples, required int sampleRate}) {
    final n = samples.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, samples);

    SherpaOnnxBindings.acceptWaveformOffline?.call(ptr, sampleRate, p, n);

    calloc.free(p);
  }

  Pointer<SherpaOnnxOfflineStream> ptr;
}
