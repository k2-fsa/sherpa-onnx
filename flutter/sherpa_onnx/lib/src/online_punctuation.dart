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

  factory OnlinePunctuationModelConfig.fromJson(Map<String, dynamic> json) {
    return OnlinePunctuationModelConfig(
      cnnBiLstm: json['cnnBiLstm'],
      bpeVocab: json['bpeVocab'],
      numThreads: json['numThreads'],
      provider: json['provider'],
      debug: json['debug'],
    );
  }

  @override
  String toString() {
    return 'OnlinePunctuationModelConfig(cnnBiLstm: $cnnBiLstm, '
        'bpeVocab: $bpeVocab, numThreads: $numThreads, '
        'provider: $provider, debug: $debug)';
  }

  Map<String, dynamic> toJson() {
    return {
      'cnnBiLstm': cnnBiLstm,
      'bpeVocab': bpeVocab,
      'numThreads': numThreads,
      'provider': provider,
      'debug': debug,
    };
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

  factory OnlinePunctuationConfig.fromJson(Map<String, dynamic> json) {
    return OnlinePunctuationConfig(
      model: OnlinePunctuationModelConfig.fromJson(json['model']),
    );
  }

  @override
  String toString() {
    return 'OnlinePunctuationConfig(model: $model)';
  }

  Map<String, dynamic> toJson() {
    return {
      'model': model.toJson(),
    };
  }

  final OnlinePunctuationModelConfig model;
}

class OnlinePunctuation {
  OnlinePunctuation.fromPtr({required this.ptr, required this.config});

  OnlinePunctuation._({required this.ptr, required this.config});

  // The user has to invoke OnlinePunctuation.free() to avoid memory leak.
  factory OnlinePunctuation({required OnlinePunctuationConfig config}) {
    if (SherpaOnnxBindings.sherpaOnnxCreateOnlinePunctuation == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final c = calloc<SherpaOnnxOnlinePunctuationConfig>();

    final cnnBiLstmPtr = config.model.cnnBiLstm.toNativeUtf8();
    final bpeVocabPtr = config.model.bpeVocab.toNativeUtf8();
    c.ref.model.cnnBiLstm = cnnBiLstmPtr;
    c.ref.model.bpeVocab = bpeVocabPtr;
    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.debug = config.model.debug ? 1 : 0;

    final providerPtr = config.model.provider.toNativeUtf8();
    c.ref.model.provider = providerPtr;

    final ptr = SherpaOnnxBindings.sherpaOnnxCreateOnlinePunctuation?.call(c) ??
        nullptr;

    calloc.free(providerPtr);
    calloc.free(cnnBiLstmPtr);
    calloc.free(bpeVocabPtr);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create online punctuation. Please check your config");
    }

    return OnlinePunctuation._(ptr: ptr, config: config);
  }

  void free() {
    SherpaOnnxBindings.sherpaOnnxDestroyOnlinePunctuation?.call(ptr);
    ptr = nullptr;
  }

  String addPunct(String text) {
    final textPtr = text.toNativeUtf8();

    final p = SherpaOnnxBindings.sherpaOnnxOnlinePunctuationAddPunct
            ?.call(ptr, textPtr) ??
        nullptr;

    calloc.free(textPtr);

    if (p == nullptr) {
      return '';
    }

    final ans = p.toDartString();

    SherpaOnnxBindings.sherpaOnnxOnlinePunctuationFreeText?.call(p);

    return ans;
  }

  Pointer<SherpaOnnxOnlinePunctuation> ptr;
  final OnlinePunctuationConfig config;
}
