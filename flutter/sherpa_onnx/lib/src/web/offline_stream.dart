// Copyright (c)  2026  Xiaomi Corporation
// Web stub for offline stream -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import 'dart:typed_data';

/// Input stream for offline APIs such as offline ASR, audio tagging, and
/// spoken language identification.
class OfflineStream {
  OfflineStream({required this.ptr});

  void free() {}
  void acceptWaveform(
      {required Float32List samples, required int sampleRate}) {}
  void setOption({required String key, required String value}) {}

  dynamic ptr;
}
