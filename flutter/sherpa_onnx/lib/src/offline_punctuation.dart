// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

class OfflinePunctuationModelConfig {
  OfflinePunctuationModelConfig(
      {required this.ctTransformer,
      this.numThreads = 1,
      this.provider = 'cpu',
      this.debug = true});

  factory OfflinePunctuationModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflinePunctuationModelConfig(
      ctTransformer: json['ctTransformer'] as String,
      numThreads: json['numThreads'] as int? ?? 1,
      provider: json['provider'] as String? ?? 'cpu',
      debug: json['debug'] as bool? ?? true,
    );
  }

  @override
  String toString() {
    return 'OfflinePunctuationModelConfig(ctTransformer: $ctTransformer, numThreads: $numThreads, provider: $provider, debug: $debug)';
  }

  Map<String, dynamic> toJson() => {
        'ctTransformer': ctTransformer,
        'numThreads': numThreads,
        'provider': provider,
        'debug': debug,
      };

  final String ctTransformer;
  final int numThreads;
  final String provider;
  final bool debug;
}

class OfflinePunctuationConfig {
  OfflinePunctuationConfig({
    required this.model,
  });

  factory OfflinePunctuationConfig.fromJson(Map<String, dynamic> json) {
    return OfflinePunctuationConfig(
      model: OfflinePunctuationModelConfig.fromJson(
          json['model'] as Map<String, dynamic>),
    );
  }

  @override
  String toString() {
    return 'OfflinePunctuationConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model.toJson(),
      };

  final OfflinePunctuationModelConfig model;
}

class OfflinePunctuation {
  OfflinePunctuation.fromPtr({required this.ptr, required this.config});

  OfflinePunctuation._({required this.ptr, required this.config});

  // The user has to invoke OfflinePunctuation.free() to avoid memory leak.
  factory OfflinePunctuation({required OfflinePunctuationConfig config}) {
    if (SherpaOnnxBindings.sherpaOnnxCreateOfflinePunctuation == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final c = calloc<SherpaOnnxOfflinePunctuationConfig>();

    final ctTransformerPtr = config.model.ctTransformer.toNativeUtf8();
    c.ref.model.ctTransformer = ctTransformerPtr;
    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.debug = config.model.debug ? 1 : 0;

    final providerPtr = config.model.provider.toNativeUtf8();
    c.ref.model.provider = providerPtr;

    final ptr =
        SherpaOnnxBindings.sherpaOnnxCreateOfflinePunctuation?.call(c) ??
            nullptr;

    calloc.free(providerPtr);
    calloc.free(ctTransformerPtr);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create offline punctuation. Please check your config");
    }

    return OfflinePunctuation._(ptr: ptr, config: config);
  }

  void free() {
    SherpaOnnxBindings.sherpaOnnxDestroyOfflinePunctuation?.call(ptr);
    ptr = nullptr;
  }

  String addPunct(String text) {
    final textPtr = text.toNativeUtf8();

    final p = SherpaOnnxBindings.sherpaOfflinePunctuationAddPunct
            ?.call(ptr, textPtr) ??
        nullptr;

    calloc.free(textPtr);

    if (p == nullptr) {
      return '';
    }

    final ans = p.toDartString();

    SherpaOnnxBindings.sherpaOfflinePunctuationFreeText?.call(p);

    return ans;
  }

  Pointer<SherpaOnnxOfflinePunctuation> ptr;
  final OfflinePunctuationConfig config;
}
