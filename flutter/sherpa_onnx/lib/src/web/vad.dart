// Copyright (c)  2026  Xiaomi Corporation
// Web implementation of VAD using dart:js_interop.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import '../vad_config.dart';
import 'init.dart';

export '../vad_config.dart';

/// Circular sample buffer used by VAD-related pipelines (web implementation).
class CircularBuffer {
  CircularBuffer.fromPtr({required this.ptr});
  CircularBuffer._({required this.ptr});

  factory CircularBuffer({required int capacity}) {
    final m = getModule();
    final cbClass =
        globalContext.getProperty('CircularBuffer'.toJS) as JSFunction?;
    if (cbClass == null) {
      throw StateError('CircularBuffer not found. Is sherpa-onnx-vad.js loaded?');
    }
    final handle = cbClass.callAsConstructor(capacity.toJS, m);
    if (handle == null) {
      throw Exception('Failed to create CircularBuffer');
    }
    return CircularBuffer._(ptr: handle);
  }

  void free() {
    if (_freed) return;
    final handle = ptr as JSObject;
    final freeFn = handle.getProperty('free'.toJS) as JSFunction?;
    freeFn?.callAsFunction(handle);
    _freed = true;
  }

  void push(Float32List data) {
    final handle = ptr as JSObject;
    final pushFn = handle.getProperty('push'.toJS) as JSFunction?;
    // Convert Float32List to JS Float32Array.
    final float32Ctor =
        globalContext.getProperty('Float32Array'.toJS) as JSFunction;
    final jsArray = float32Ctor.callAsConstructor(data.buffer.toJS);
    pushFn?.callAsFunction(handle, jsArray);
  }

  Float32List get({required int startIndex, required int n}) {
    final handle = ptr as JSObject;
    final getFn = handle.getProperty('get'.toJS) as JSFunction?;
    if (getFn == null) return Float32List(0);
    final result = getFn.callAsFunction(handle, startIndex.toJS, n.toJS);
    if (result == null) return Float32List(0);
    return (result as JSFloat32Array).toDart;
  }

  void pop(int n) {
    final handle = ptr as JSObject;
    final popFn = handle.getProperty('pop'.toJS) as JSFunction?;
    popFn?.callAsFunction(handle, n.toJS);
  }

  void reset() {
    final handle = ptr as JSObject;
    final resetFn = handle.getProperty('reset'.toJS) as JSFunction?;
    resetFn?.callAsFunction(handle);
  }

  int get size {
    final handle = ptr as JSObject;
    final sizeFn = handle.getProperty('size'.toJS) as JSFunction?;
    if (sizeFn == null) return 0;
    final result = sizeFn.callAsFunction(handle);
    return (result as JSNumber).toDartInt;
  }

  int get head {
    final handle = ptr as JSObject;
    final headFn = handle.getProperty('head'.toJS) as JSFunction?;
    if (headFn == null) return 0;
    final result = headFn.callAsFunction(handle);
    return (result as JSNumber).toDartInt;
  }

  dynamic ptr;
  bool _freed = false;
}

/// Voice activity detector that emits [SpeechSegment] objects (web implementation).
class VoiceActivityDetector {
  VoiceActivityDetector.fromPtr({required this.ptr, required this.config});
  VoiceActivityDetector._({required this.ptr, required this.config});

  factory VoiceActivityDetector(
      {required VadModelConfig config, required double bufferSizeInSeconds}) {
    final m = getModule();

    // Use createVad from sherpa-onnx-vad.js.
    final createFn =
        globalContext.getProperty('createVad'.toJS) as JSFunction?;
    if (createFn == null) {
      throw StateError('createVad not found. Is sherpa-onnx-vad.js loaded?');
    }

    // Build config JSON.
    final jsConfig = JSObject();

    final sileroVad = JSObject();
    sileroVad['model'] = config.sileroVad.model.toJS;
    sileroVad['threshold'] = config.sileroVad.threshold.toJS;
    sileroVad['minSilenceDuration'] = config.sileroVad.minSilenceDuration.toJS;
    sileroVad['minSpeechDuration'] = config.sileroVad.minSpeechDuration.toJS;
    sileroVad['windowSize'] = config.sileroVad.windowSize.toJS;
    sileroVad['maxSpeechDuration'] = config.sileroVad.maxSpeechDuration.toJS;
    jsConfig['sileroVad'] = sileroVad;

    final tenVad = JSObject();
    tenVad['model'] = config.tenVad.model.toJS;
    tenVad['threshold'] = config.tenVad.threshold.toJS;
    tenVad['minSilenceDuration'] = config.tenVad.minSilenceDuration.toJS;
    tenVad['minSpeechDuration'] = config.tenVad.minSpeechDuration.toJS;
    tenVad['windowSize'] = config.tenVad.windowSize.toJS;
    tenVad['maxSpeechDuration'] = config.tenVad.maxSpeechDuration.toJS;
    jsConfig['tenVad'] = tenVad;

    jsConfig['sampleRate'] = config.sampleRate.toJS;
    jsConfig['numThreads'] = config.numThreads.toJS;
    jsConfig['provider'] = config.provider.toJS;
    jsConfig['debug'] = config.debug.toJS;
    jsConfig['bufferSizeInSeconds'] = bufferSizeInSeconds.toJS;

    final handle = createFn.callAsFunction(null, m, jsConfig);
    if (handle == null) {
      throw Exception('Failed to create VoiceActivityDetector');
    }

    return VoiceActivityDetector._(ptr: handle, config: config);
  }

  void free() {
    if (_freed) return;
    final handle = ptr as JSObject;
    final freeFn = handle.getProperty('free'.toJS) as JSFunction?;
    freeFn?.callAsFunction(handle);
    _freed = true;
  }

  void acceptWaveform(Float32List samples) {
    final handle = ptr as JSObject;
    final fn = handle.getProperty('acceptWaveform'.toJS) as JSFunction?;
    // Convert Float32List to JS Float32Array.
    final float32Ctor =
        globalContext.getProperty('Float32Array'.toJS) as JSFunction;
    final jsArray = float32Ctor.callAsConstructor(samples.buffer.toJS);
    fn?.callAsFunction(handle, jsArray);
  }

  bool isEmpty() {
    final handle = ptr as JSObject;
    final fn = handle.getProperty('isEmpty'.toJS) as JSFunction?;
    if (fn == null) return true;
    final result = fn.callAsFunction(handle);
    return (result as JSBoolean).toDart;
  }

  bool isDetected() {
    final handle = ptr as JSObject;
    final fn = handle.getProperty('isDetected'.toJS) as JSFunction?;
    if (fn == null) return false;
    final result = fn.callAsFunction(handle);
    return (result as JSBoolean).toDart;
  }

  void pop() {
    final handle = ptr as JSObject;
    final fn = handle.getProperty('pop'.toJS) as JSFunction?;
    fn?.callAsFunction(handle);
  }

  void clear() {
    final handle = ptr as JSObject;
    final fn = handle.getProperty('clear'.toJS) as JSFunction?;
    fn?.callAsFunction(handle);
  }

  SpeechSegment front() {
    final handle = ptr as JSObject;
    final fn = handle.getProperty('front'.toJS) as JSFunction?;
    if (fn == null) return SpeechSegment(samples: Float32List(0), start: 0);

    final result = fn.callAsFunction(handle) as JSObject;
    final samples = (result.getProperty('samples'.toJS) as JSFloat32Array).toDart;
    final start = (result.getProperty('start'.toJS) as JSNumber).toDartInt;

    return SpeechSegment(samples: Float32List.fromList(samples), start: start);
  }

  void reset() {
    final handle = ptr as JSObject;
    final fn = handle.getProperty('reset'.toJS) as JSFunction?;
    fn?.callAsFunction(handle);
  }

  void flush() {
    final handle = ptr as JSObject;
    final fn = handle.getProperty('flush'.toJS) as JSFunction?;
    fn?.callAsFunction(handle);
  }

  dynamic ptr;
  final VadModelConfig config;
  bool _freed = false;
}
