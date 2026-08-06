// Copyright (c)  2026  Xiaomi Corporation
// Web stub for spoken language identification -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import 'offline_stream.dart';
import '../spoken_language_identification_config.dart';

export '../spoken_language_identification_config.dart';

/// Spoken language identifier.
class SpokenLanguageIdentification {
  SpokenLanguageIdentification.fromPtr(
      {required this.ptr, required this.config});
  SpokenLanguageIdentification._({required this.ptr, required this.config});

  factory SpokenLanguageIdentification(
      SpokenLanguageIdentificationConfig config) {
    throw UnsupportedError(
        'SpokenLanguageIdentification is not yet supported on web');
  }

  void free() {}
  OfflineStream createStream() =>
      throw UnsupportedError(
          'SpokenLanguageIdentification is not yet supported on web');
  SpokenLanguageIdentificationResult compute(OfflineStream stream) =>
      const SpokenLanguageIdentificationResult(lang: '');

  dynamic ptr;
  SpokenLanguageIdentificationConfig config;
}
