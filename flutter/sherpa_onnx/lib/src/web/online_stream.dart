// Copyright (c)  2026  Xiaomi Corporation
// Web stub for online stream -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import 'dart:typed_data';

/// Input stream for streaming APIs such as online ASR and keyword spotting.
class OnlineStream {
  OnlineStream({required this.ptr});

  void free() {}
  void acceptWaveform(
      {required Float32List samples, required int sampleRate}) {}
  void inputFinished() {}
  void setOption({required String key, required String value}) {}

  dynamic ptr;
}
