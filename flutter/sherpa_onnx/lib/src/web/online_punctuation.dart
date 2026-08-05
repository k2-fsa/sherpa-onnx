// Copyright (c)  2026  Xiaomi Corporation
// Web stub for online punctuation -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import '../online_punctuation_config.dart';

export '../online_punctuation_config.dart';

/// Online punctuation restorer.
class OnlinePunctuation {
  OnlinePunctuation.fromPtr({required this.ptr, required this.config});
  OnlinePunctuation._({required this.ptr, required this.config});

  factory OnlinePunctuation({required OnlinePunctuationConfig config}) {
    throw UnsupportedError('OnlinePunctuation is not yet supported on web');
  }

  void free() {}
  String addPunct(String text) => '';

  dynamic ptr;
  final OnlinePunctuationConfig config;
}
