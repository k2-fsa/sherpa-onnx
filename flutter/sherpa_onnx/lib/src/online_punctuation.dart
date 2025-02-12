import 'dart:ffi';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

class OnlinePunctuationModelConfig {
  OnlinePunctuationModelConfig(
      {required this.cnnBiLstm,
      required this.bpeVocab,
      this.numThreads = 1,
      this.provider = 'cpu',
      this.debug = true});

  @override
  String toString() {
    return 'OnlinePunctuationModelConfig(cnnBiLstm: $cnnBiLstm, bpeVocab: $bpeVocab, numThreads: $numThreads, provider: $provider, debug: $debug)';
  }

  final String cnnBiLstm;
  final String bpeVocab;
  final int numThreads;
  final String provider;
  final bool debug;
}

class OnlinePunctuationConfig {
  OnlinePunctuationConfig({
    required this.model,
  });

  @override
  String toString() {
    return 'OnlinePunctuationConfig(model: $model)';
  }

  final OnlinePunctuationModelConfig model;
}

class OnlinePunctuation {
  OnlinePunctuation.fromPtr({required this.ptr, required this.config});
  OnlinePunctuation._({required this.ptr, required this.config});

  factory OnlinePunctuation({required OnlinePunctuationConfig config}) {
    final c = calloc<SherpaOnnxOnlinePunctuationConfig>();

    final cnnBiLstmPtr = config.model.cnnBiLstm.toNativeUtf8();
    final bpeVocabPtr = config.model.bpeVocab.toNativeUtf8();

    c.ref.model.cnnBiLstm = cnnBiLstmPtr;
    c.ref.model.bpeVocab = bpeVocabPtr;
    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.debug = config.model.debug ? 1 : 0;
    c.ref.model.provider = config.model.provider.toNativeUtf8();

    final ptr = SherpaOnnxBindings.sherpaOnnxCreateOnlinePunctuation?.call(c) ??
        nullptr;

    // Free allocated memory
    calloc.free(cnnBiLstmPtr);
    calloc.free(bpeVocabPtr);
    calloc.free(c);

    return OnlinePunctuation._(ptr: ptr, config: config);
  }

  void free() {
    SherpaOnnxBindings.sherpaOnnxDestroyOnlinePunctuation?.call(ptr);
    ptr = nullptr;
  }

  String addPunct(String text) {
    final textPtr = text.toNativeUtf8();
    final p = SherpaOnnxBindings.sherpaOnlinePunctuationAddPunct
            ?.call(ptr, textPtr) ??
        nullptr;
    calloc.free(textPtr);

    if (p == nullptr) return '';

    final ans = p.toDartString();
    SherpaOnnxBindings.sherpaOnlinePunctuationFreeText?.call(p);
    return ans;
  }

  Pointer<SherpaOnnxOnlinePunctuation> ptr;
  final OnlinePunctuationConfig config;
}
