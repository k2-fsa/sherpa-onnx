// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';
import './offline_punctuation_config.dart';

export './offline_punctuation_config.dart';

/// Offline punctuation restorer.
class OfflinePunctuation {
  OfflinePunctuation.fromPtr({required this.ptr, required this.config});

  OfflinePunctuation._({required this.ptr, required this.config});

  /// Create an offline punctuator from [config].
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

  /// Release the native punctuator.
  void free() {
    if (SherpaOnnxBindings.sherpaOnnxDestroyOfflinePunctuation == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.sherpaOnnxDestroyOfflinePunctuation?.call(ptr);
    ptr = nullptr;
  }

  /// Add punctuation to [text].
  String addPunct(String text) {
    if (SherpaOnnxBindings.sherpaOfflinePunctuationAddPunct == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return '';
    }

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
