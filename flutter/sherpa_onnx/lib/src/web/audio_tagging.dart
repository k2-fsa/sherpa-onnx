// Copyright (c)  2026  Xiaomi Corporation
// Web stub for audio tagging -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import '../audio_tagging_config.dart';
import 'offline_stream.dart';

export '../audio_tagging_config.dart';

/// Offline audio tagger.
class AudioTagging {
  AudioTagging.fromPtr({required this.ptr, required this.config});
  AudioTagging._({required this.ptr, required this.config});

  factory AudioTagging({required AudioTaggingConfig config}) {
    throw UnsupportedError('AudioTagging is not yet supported on web');
  }

  void free() {}
  OfflineStream createStream() =>
      throw UnsupportedError('AudioTagging is not yet supported on web');
  List<AudioEvent> compute(
          {required OfflineStream stream, required int topK}) =>
      <AudioEvent>[];

  dynamic ptr;
  final AudioTaggingConfig config;
}
