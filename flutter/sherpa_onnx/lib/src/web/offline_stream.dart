// Copyright (c)  2026  Xiaomi Corporation
// Web implementation of OfflineStream using dart:js_interop.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

/// Input stream for offline APIs such as offline ASR, audio tagging, and
/// spoken language identification.
class OfflineStream {
  OfflineStream({required this.ptr});

  final dynamic ptr;
  bool _freed = false;

  void free() {
    if (_freed) return;
    final handle = ptr as JSObject;
    final freeFn = handle.getProperty('free'.toJS) as JSFunction?;
    freeFn?.callAsFunction(handle);
    _freed = true;
  }

  void acceptWaveform(
      {required Float32List samples, required int sampleRate}) {
    final handle = ptr as JSObject;
    final acceptFn =
        handle.getProperty('acceptWaveform'.toJS) as JSFunction?;

    // Convert Float32List to JS Float32Array.
    // Same approach as VAD's CircularBuffer.push().
    final float32Ctor =
        globalContext.getProperty('Float32Array'.toJS) as JSFunction;
    final jsArray = float32Ctor.callAsConstructor(samples.buffer.toJS);

    // JS API: acceptWaveform(sampleRate, samples)
    acceptFn?.callAsFunction(handle, sampleRate.toJS, jsArray);
  }

  void setOption({required String key, required String value}) {
    final handle = ptr as JSObject;
    final setOptionFn = handle.getProperty('setOption'.toJS) as JSFunction?;
    setOptionFn?.callAsFunction(handle, key.toJS, value.toJS);
  }

  String getOption({required String key}) {
    final handle = ptr as JSObject;
    final getOptionFn = handle.getProperty('getOption'.toJS) as JSFunction?;
    if (getOptionFn == null) return '';
    final result = getOptionFn.callAsFunction(handle, key.toJS);
    return (result as JSString?)?.toDart ?? '';
  }
}
