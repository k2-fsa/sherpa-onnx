// Copyright (c)  2026  Xiaomi Corporation
// Web stub for offline recognizer -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import '../offline_recognizer_config.dart';
import 'offline_stream.dart';

export '../offline_recognizer_config.dart';

/// Offline speech recognizer.
///
/// Create one from an [OfflineRecognizerConfig], then create an
/// [OfflineStream], feed waveform samples, call [decode], and fetch the final
/// hypothesis with [getResult].
class OfflineRecognizer {
  OfflineRecognizer.fromPtr({required this.ptr, required this.config});

  factory OfflineRecognizer(OfflineRecognizerConfig config) {
    throw UnsupportedError('OfflineRecognizer is not yet supported on web');
  }

  void free() {}
  void setConfig(OfflineRecognizerConfig config) {}
  OfflineStream createStream() =>
      throw UnsupportedError('OfflineRecognizer is not yet supported on web');
  void decode(OfflineStream stream) {}
  OfflineRecognizerResult getResult(OfflineStream stream) =>
      OfflineRecognizerResult(
        text: '',
        tokens: [],
        timestamps: [],
        lang: '',
        emotion: '',
        event: '',
      );

  dynamic ptr;
  OfflineRecognizerConfig config;
}
