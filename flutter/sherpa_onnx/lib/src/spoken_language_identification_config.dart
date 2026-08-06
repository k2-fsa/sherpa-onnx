// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for spoken language identification -- no FFI, works on all platforms.

/// Model files for spoken language identification using Whisper.
class SpokenLanguageIdentificationWhisperConfig {
  const SpokenLanguageIdentificationWhisperConfig({
    this.encoder = '',
    this.decoder = '',
    this.tailPaddings = 0,
  });

  factory SpokenLanguageIdentificationWhisperConfig.fromJson(
      Map<String, dynamic> json) {
    return SpokenLanguageIdentificationWhisperConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      tailPaddings: json['tailPaddings'] as int? ?? 0,
    );
  }

  @override
  String toString() {
    return 'SpokenLanguageIdentificationWhisperConfig(encoder: $encoder, decoder: $decoder, tailPaddings: $tailPaddings)';
  }

  Map<String, dynamic> toJson() => {
        'encoder': encoder,
        'decoder': decoder,
        'tailPaddings': tailPaddings,
      };

  final String encoder;
  final String decoder;
  final int tailPaddings;
}

/// Top-level configuration for [SpokenLanguageIdentification].
class SpokenLanguageIdentificationConfig {
  const SpokenLanguageIdentificationConfig({
    this.whisper = const SpokenLanguageIdentificationWhisperConfig(),
    this.numThreads = 1,
    this.debug = false,
    this.provider = 'cpu',
  });

  factory SpokenLanguageIdentificationConfig.fromJson(
      Map<String, dynamic> json) {
    return SpokenLanguageIdentificationConfig(
      whisper: json['whisper'] != null
          ? SpokenLanguageIdentificationWhisperConfig.fromJson(
              json['whisper'] as Map<String, dynamic>)
          : const SpokenLanguageIdentificationWhisperConfig(),
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? false,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'SpokenLanguageIdentificationConfig(whisper: $whisper, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'whisper': whisper.toJson(),
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final SpokenLanguageIdentificationWhisperConfig whisper;
  final int numThreads;
  final bool debug;
  final String provider;
}

/// Result returned by [SpokenLanguageIdentification.compute].
class SpokenLanguageIdentificationResult {
  const SpokenLanguageIdentificationResult({
    required this.lang,
  });

  factory SpokenLanguageIdentificationResult.fromJson(
      Map<String, dynamic> json) {
    return SpokenLanguageIdentificationResult(
      lang: json['lang'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'SpokenLanguageIdentificationResult(lang: $lang)';
  }

  Map<String, dynamic> toJson() => {
        'lang': lang,
      };

  final String lang;
}
