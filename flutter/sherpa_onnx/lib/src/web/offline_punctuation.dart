// Copyright (c)  2026  Xiaomi Corporation
// Web stub for offline punctuation -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import '../offline_punctuation_config.dart';

export '../offline_punctuation_config.dart';

/// Offline punctuation restorer.
class OfflinePunctuation {
  OfflinePunctuation.fromPtr({required this.ptr, required this.config});
  OfflinePunctuation._({required this.ptr, required this.config});

  factory OfflinePunctuation({required OfflinePunctuationConfig config}) {
    throw UnsupportedError('OfflinePunctuation is not yet supported on web');
  }

  void free() {}
  String addPunct(String text) => '';

  dynamic ptr;
  final OfflinePunctuationConfig config;
}
