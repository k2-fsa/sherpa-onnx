// Copyright (c)  2026  Xiaomi Corporation
// Web stub for online recognizer -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import '../online_recognizer_config.dart';
import 'online_stream.dart';

export '../online_recognizer_config.dart';

/// Streaming speech recognizer.
///
/// Create one from an [OnlineRecognizerConfig], then feed chunks to an
/// [OnlineStream] and call [decode] while [isReady] is true.
class OnlineRecognizer {
  OnlineRecognizer.fromPtr({required this.ptr, required this.config});
  OnlineRecognizer._({required this.ptr, required this.config});

  factory OnlineRecognizer(OnlineRecognizerConfig config) {
    throw UnsupportedError('OnlineRecognizer is not yet supported on web');
  }

  void free() {}
  OnlineStream createStream({String hotwords = ''}) =>
      throw UnsupportedError('OnlineRecognizer is not yet supported on web');
  bool isReady(OnlineStream stream) => false;
  OnlineRecognizerResult getResult(OnlineStream stream) =>
      OnlineRecognizerResult(text: '', tokens: [], timestamps: []);
  void reset(OnlineStream stream) {}
  void decode(OnlineStream stream) {}
  bool isEndpoint(OnlineStream stream) => false;

  dynamic ptr;
  OnlineRecognizerConfig config;
}
