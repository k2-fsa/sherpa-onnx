// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

/// Input stream for streaming APIs such as online ASR and keyword spotting.
class OnlineStream {
  /// The user has to call OnlineStream.free() to avoid memory leak.
  OnlineStream({required this.ptr});

  /// Release the native stream.
  void free() {
    if (SherpaOnnxBindings.destroyOnlineStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroyOnlineStream?.call(ptr);
    ptr = nullptr;
  }

  /// If you have List<double> data, then you can use
  /// Float32List.fromList(data) to convert data to Float32List
  ///
  /// See
  ///  https://api.flutter.dev/flutter/dart-core/List-class.html
  /// and
  ///  https://api.flutter.dev/flutter/dart-typed_data/Float32List-class.html
  /// Append waveform samples to the stream.
  ///
  /// [samples] must contain mono floating-point PCM data normalized to
  /// `[-1, 1]`. Feed your audio in chunks, then call [inputFinished] after the
  /// last chunk if you want the recognizer to flush trailing context.
  void acceptWaveform({required Float32List samples, required int sampleRate}) {
    if (SherpaOnnxBindings.onlineStreamAcceptWaveform == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }

    final n = samples.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, samples);

    SherpaOnnxBindings.onlineStreamAcceptWaveform?.call(ptr, sampleRate, p, n);

    calloc.free(p);
  }

  /// Mark the end of input.
  void inputFinished() {
    if (SherpaOnnxBindings.onlineStreamInputFinished == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.onlineStreamInputFinished?.call(ptr);
  }

  Pointer<SherpaOnnxOnlineStream> ptr;
}
