/// Shared config/data classes for online punctuation -- no FFI, works on all platforms.

/// Online punctuation restoration.
///
/// This wrapper is intended for shorter or incremental text fragments. See
/// `dart-api-examples/add-punctuations/` for working examples.
class OnlinePunctuationModelConfig {
  OnlinePunctuationModelConfig(
      {required this.cnnBiLstm,
      required this.bpeVocab,
      this.numThreads = 1,
      this.provider = 'cpu',
      this.debug = true});

  factory OnlinePunctuationModelConfig.fromJson(Map<String, dynamic> json) {
    return OnlinePunctuationModelConfig(
      cnnBiLstm: json['cnnBiLstm'],
      bpeVocab: json['bpeVocab'],
      numThreads: json['numThreads'],
      provider: json['provider'],
      debug: json['debug'],
    );
  }

  @override
  String toString() {
    return 'OnlinePunctuationModelConfig(cnnBiLstm: $cnnBiLstm, '
        'bpeVocab: $bpeVocab, numThreads: $numThreads, '
        'provider: $provider, debug: $debug)';
  }

  Map<String, dynamic> toJson() {
    return {
      'cnnBiLstm': cnnBiLstm,
      'bpeVocab': bpeVocab,
      'numThreads': numThreads,
      'provider': provider,
      'debug': debug,
    };
  }

  final String cnnBiLstm;
  final String bpeVocab;
  final int numThreads;
  final String provider;
  final bool debug;
}

/// Top-level configuration for [OnlinePunctuation].
class OnlinePunctuationConfig {
  OnlinePunctuationConfig({
    required this.model,
  });

  factory OnlinePunctuationConfig.fromJson(Map<String, dynamic> json) {
    return OnlinePunctuationConfig(
      model: OnlinePunctuationModelConfig.fromJson(json['model']),
    );
  }

  @override
  String toString() {
    return 'OnlinePunctuationConfig(model: $model)';
  }

  Map<String, dynamic> toJson() {
    return {
      'model': model.toJson(),
    };
  }

  final OnlinePunctuationModelConfig model;
}
