// Copyright (c)  2026  Xiaomi Corporation
// Web stub for keyword spotter -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import '../keyword_spotter_config.dart';
import 'online_stream.dart';

export '../keyword_spotter_config.dart';

/// Streaming keyword spotter.
class KeywordSpotter {
  KeywordSpotter.fromPtr({required this.ptr, required this.config});
  KeywordSpotter._({required this.ptr, required this.config});

  factory KeywordSpotter(KeywordSpotterConfig config) {
    throw UnsupportedError('KeywordSpotter is not yet supported on web');
  }

  void free() {}
  OnlineStream createStream({String keywords = ''}) =>
      throw UnsupportedError('KeywordSpotter is not yet supported on web');
  bool isReady(OnlineStream stream) => false;
  KeywordResult getResult(OnlineStream stream) => KeywordResult(keyword: '');
  void decode(OnlineStream stream) {}
  void reset(OnlineStream stream) {}

  dynamic ptr;
  KeywordSpotterConfig config;
}
