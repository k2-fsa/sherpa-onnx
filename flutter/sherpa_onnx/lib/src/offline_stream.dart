// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

/// Input stream for offline APIs such as offline ASR, audio tagging, and
/// spoken language identification.
class OfflineStream {
  /// The user has to call OfflineStream.free() to avoid memory leak.
  OfflineStream({required this.ptr});

  /// Release the native stream.
  void free() {
    if (SherpaOnnxBindings.destroyOfflineStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
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
  /// Append waveform samples to the stream.
  ///
  /// [samples] must contain mono floating-point PCM data normalized to
  /// `[-1, 1]`. [sampleRate] should match the model expectation, typically
  /// 16000 for the provided examples.
  void acceptWaveform({required Float32List samples, required int sampleRate}) {
    if (SherpaOnnxBindings.acceptWaveformOffline == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }

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

    // Defensive validation for native values
    if (numTokens < 0 ||
        vocabSize < 0 ||
        numTokens > 10000 ||
        vocabSize > 100000) {
      destroyFunc(vocabPtr);
      return null;
    }

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
