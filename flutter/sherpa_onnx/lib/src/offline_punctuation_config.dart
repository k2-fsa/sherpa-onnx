// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for offline punctuation -- no FFI, works on all platforms.

/// Offline punctuation restoration.
///
/// This is intended for complete text strings when you want one-shot
/// punctuation insertion. See `dart-api-examples/add-punctuations/`.
class OfflinePunctuationModelConfig {
  OfflinePunctuationModelConfig(
      {required this.ctTransformer,
      this.numThreads = 1,
      this.provider = 'cpu',
      this.debug = true});

  factory OfflinePunctuationModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflinePunctuationModelConfig(
      ctTransformer: json['ctTransformer'] as String,
      numThreads: json['numThreads'] as int? ?? 1,
      provider: json['provider'] as String? ?? 'cpu',
      debug: json['debug'] as bool? ?? true,
    );
  }

  @override
  String toString() {
    return 'OfflinePunctuationModelConfig(ctTransformer: $ctTransformer, numThreads: $numThreads, provider: $provider, debug: $debug)';
  }

  Map<String, dynamic> toJson() => {
        'ctTransformer': ctTransformer,
        'numThreads': numThreads,
        'provider': provider,
        'debug': debug,
      };

  final String ctTransformer;
  final int numThreads;
  final String provider;
  final bool debug;
}

/// Top-level configuration for [OfflinePunctuation].
class OfflinePunctuationConfig {
  OfflinePunctuationConfig({
    required this.model,
  });

  factory OfflinePunctuationConfig.fromJson(Map<String, dynamic> json) {
    return OfflinePunctuationConfig(
      model: OfflinePunctuationModelConfig.fromJson(
          json['model'] as Map<String, dynamic>),
    );
  }

  @override
  String toString() {
    return 'OfflinePunctuationConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model.toJson(),
      };

  final OfflinePunctuationModelConfig model;
}
