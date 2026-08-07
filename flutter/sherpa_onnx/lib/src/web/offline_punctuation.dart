// Copyright (c)  2026  Xiaomi Corporation
// Web implementation of OfflinePunctuation using dart:js_interop.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import '../offline_punctuation_config.dart';
import 'init.dart';

export '../offline_punctuation_config.dart';

/// Offline punctuation restorer (web implementation).
class OfflinePunctuation {
  OfflinePunctuation.fromPtr({required this.ptr, required this.config});
  OfflinePunctuation._({required this.ptr, required this.config});

  /// Create an offline punctuator from [config].
  factory OfflinePunctuation({required OfflinePunctuationConfig config}) {
    final m = getModule();

    // Use the initSherpaOnnxOfflinePunctuationConfig helper from sherpa-onnx-punctuation.js.
    final initFn = globalContext.getProperty(
        'initSherpaOnnxOfflinePunctuationConfig'.toJS) as JSFunction?;
    if (initFn == null) {
      throw StateError(
          'initSherpaOnnxOfflinePunctuationConfig not found. '
          'Is sherpa-onnx-punctuation.js loaded?');
    }

    // Build config JSON.
    final jsConfig = JSObject();
    final model = JSObject();
    model['ctTransformer'] = config.model.ctTransformer.toJS;
    model['numThreads'] = config.model.numThreads.toJS;
    model['debug'] = config.model.debug.toJS;
    model['provider'] = config.model.provider.toJS;
    jsConfig['model'] = model;

    // Allocate native config.
    final wasmConfig = initFn.callAsFunction(null, jsConfig, m) as JSObject;
    final configPtr = wasmConfig.getProperty('ptr'.toJS);

    // Create punctuation instance.
    final createFn = m.getProperty('_SherpaOnnxCreateOfflinePunctuation'.toJS)
        as JSFunction?;
    if (createFn == null) {
      throw StateError('SherpaOnnxCreateOfflinePunctuation not found in WASM');
    }

    final handle = createFn.callAsFunction(null, configPtr);

    // Free config.
    final freeConfigFn =
        globalContext.getProperty('freeConfig'.toJS) as JSFunction?;
    freeConfigFn?.callAsFunction(null, wasmConfig, m);

    if (handle == null || (handle is JSNumber && handle.toDartDouble == 0)) {
      throw Exception('Failed to create OfflinePunctuation');
    }

    return OfflinePunctuation._(ptr: handle, config: config);
  }

  /// Release the native punctuator.
  void free() {
    if (_freed) return;
    final m = getModule();
    final destroyFn = m.getProperty('_SherpaOnnxDestroyOfflinePunctuation'.toJS)
        as JSFunction?;
    destroyFn?.callAsFunction(null, ptr);
    _freed = true;
  }

  /// Add punctuation to [text].
  String addPunct(String text) {
    final m = getModule();
    final addPunctFn =
        m.getProperty('_SherpaOfflinePunctuationAddPunct'.toJS) as JSFunction?;
    if (addPunctFn == null) return '';

    final lengthBytesUTF8 =
        m.getProperty('lengthBytesUTF8'.toJS) as JSFunction;
    final malloc = m.getProperty('_malloc'.toJS) as JSFunction;
    final stringToUTF8 =
        m.getProperty('stringToUTF8'.toJS) as JSFunction;
    final freeFn = m.getProperty('_free'.toJS) as JSFunction;
    final utf8ToString =
        m.getProperty('UTF8ToString'.toJS) as JSFunction;

    final textLen =
        (lengthBytesUTF8.callAsFunction(m, text.toJS) as JSNumber).toDartInt +
            1;
    final textPtr = malloc.callAsFunction(null, textLen.toJS);
    stringToUTF8.callAsFunction(m, text.toJS, textPtr, textLen.toJS);

    final resultPtr = addPunctFn.callAsFunction(null, ptr, textPtr);
    freeFn.callAsFunction(null, textPtr);

    if (resultPtr == null ||
        (resultPtr is JSNumber && resultPtr.toDartDouble == 0)) {
      return '';
    }

    final result =
        (utf8ToString.callAsFunction(m, resultPtr) as JSString).toDart;

    final freeTextFn =
        m.getProperty('_SherpaOfflinePunctuationFreeText'.toJS) as JSFunction?;
    freeTextFn?.callAsFunction(null, resultPtr);

    return result;
  }

  dynamic ptr;
  final OfflinePunctuationConfig config;
  bool _freed = false;
}
