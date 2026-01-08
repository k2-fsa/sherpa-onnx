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

  Map<String, List<double>>? getVocabLogProbs() {
    final getFunc = SherpaOnnxBindings.getOfflineStreamVocabLogProbs;
    final destroyFunc = SherpaOnnxBindings.destroyVocabLogProbs;

    if (getFunc == null || destroyFunc == null) {
      return null;
    }

    final vocabPtr = getFunc(ptr);
    if (vocabPtr == nullptr) {
      return null;
    }

    final vocabLogProbs = vocabPtr.ref;
    final numTokens = vocabLogProbs.numTokens;
    final vocabSize = vocabLogProbs.vocabSize;

    final Map<String, List<double>> result = {};

    for (int tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
      final List<double> tokenProbs = [];

      for (int vocabIdx = 0; vocabIdx < vocabSize; vocabIdx++) {
        final index = tokenIdx * vocabSize + vocabIdx;
        final logProb = vocabLogProbs.logProbs[index];
        tokenProbs.add(logProb);
      }

      result['token_$tokenIdx'] = tokenProbs;
    }

    destroyFunc(vocabPtr);
    return result;
  }

  Pointer<SherpaOnnxOfflineStream> ptr;
}
