// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for online recognition -- no FFI, works on all platforms.

import './feature_config.dart';
import './homophone_replacer_config.dart';

/// Model files for a streaming transducer recognizer.
class OnlineTransducerModelConfig {
  const OnlineTransducerModelConfig({
    this.encoder = '',
    this.decoder = '',
    this.joiner = '',
  });

  factory OnlineTransducerModelConfig.fromJson(Map<String, dynamic> json) {
    return OnlineTransducerModelConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      joiner: json['joiner'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OnlineTransducerModelConfig(encoder: $encoder, decoder: $decoder, joiner: $joiner)';
  }

  Map<String, dynamic> toJson() => {
        'encoder': encoder,
        'decoder': decoder,
        'joiner': joiner,
      };

  final String encoder;
  final String decoder;
  final String joiner;
}

/// Model files for a streaming Paraformer recognizer.
class OnlineParaformerModelConfig {
  const OnlineParaformerModelConfig({this.encoder = '', this.decoder = ''});

  factory OnlineParaformerModelConfig.fromJson(Map<String, dynamic> json) {
    return OnlineParaformerModelConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OnlineParaformerModelConfig(encoder: $encoder, decoder: $decoder)';
  }

  Map<String, dynamic> toJson() => {
        'encoder': encoder,
        'decoder': decoder,
      };

  final String encoder;
  final String decoder;
}

/// Model file for a streaming Zipformer2 CTC recognizer.
class OnlineZipformer2CtcModelConfig {
  const OnlineZipformer2CtcModelConfig({this.model = ''});

  factory OnlineZipformer2CtcModelConfig.fromJson(Map<String, dynamic> json) {
    return OnlineZipformer2CtcModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OnlineZipformer2CtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

/// Model file for a streaming NeMo CTC recognizer.
class OnlineNemoCtcModelConfig {
  const OnlineNemoCtcModelConfig({this.model = ''});

  factory OnlineNemoCtcModelConfig.fromJson(Map<String, dynamic> json) {
    return OnlineNemoCtcModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OnlineNemoCtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

/// Model file for a streaming tone-aware CTC recognizer.
class OnlineToneCtcModelConfig {
  const OnlineToneCtcModelConfig({this.model = ''});

  factory OnlineToneCtcModelConfig.fromJson(Map<String, dynamic> json) {
    return OnlineToneCtcModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OnlineToneCtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

/// Aggregate model configuration for streaming recognition.
///
/// Configure exactly one model family for a typical deployment and supply the
/// shared tokenizer and runtime settings here.
class OnlineModelConfig {
  const OnlineModelConfig({
    this.transducer = const OnlineTransducerModelConfig(),
    this.paraformer = const OnlineParaformerModelConfig(),
    this.zipformer2Ctc = const OnlineZipformer2CtcModelConfig(),
    this.nemoCtc = const OnlineNemoCtcModelConfig(),
    this.toneCtc = const OnlineToneCtcModelConfig(),
    required this.tokens,
    this.numThreads = 1,
    this.provider = 'cpu',
    this.debug = true,
    this.modelType = '',
    this.modelingUnit = '',
    this.bpeVocab = '',
  });

  factory OnlineModelConfig.fromJson(Map<String, dynamic> json) {
    return OnlineModelConfig(
      transducer: OnlineTransducerModelConfig.fromJson(
          json['transducer'] as Map<String, dynamic>? ?? const {}),
      paraformer: OnlineParaformerModelConfig.fromJson(
          json['paraformer'] as Map<String, dynamic>? ?? const {}),
      zipformer2Ctc: OnlineZipformer2CtcModelConfig.fromJson(
          json['zipformer2Ctc'] as Map<String, dynamic>? ?? const {}),
      nemoCtc: OnlineNemoCtcModelConfig.fromJson(
          json['nemoCtc'] as Map<String, dynamic>? ?? const {}),
      toneCtc: OnlineToneCtcModelConfig.fromJson(
          json['toneCtc'] as Map<String, dynamic>? ?? const {}),
      tokens: json['tokens'] as String,
      numThreads: json['numThreads'] as int? ?? 1,
      provider: json['provider'] as String? ?? 'cpu',
      debug: json['debug'] as bool? ?? true,
      modelType: json['modelType'] as String? ?? '',
      modelingUnit: json['modelingUnit'] as String? ?? '',
      bpeVocab: json['bpeVocab'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OnlineModelConfig(transducer: $transducer, paraformer: $paraformer, zipformer2Ctc: $zipformer2Ctc, nemoCtc: $nemoCtc, toneCtc: $toneCtc, tokens: $tokens, numThreads: $numThreads, provider: $provider, debug: $debug, modelType: $modelType, modelingUnit: $modelingUnit, bpeVocab: $bpeVocab)';
  }

  Map<String, dynamic> toJson() => {
        'transducer': transducer.toJson(),
        'paraformer': paraformer.toJson(),
        'zipformer2Ctc': zipformer2Ctc.toJson(),
        'nemoCtc': nemoCtc.toJson(),
        'toneCtc': toneCtc.toJson(),
        'tokens': tokens,
        'numThreads': numThreads,
        'provider': provider,
        'debug': debug,
        'modelType': modelType,
        'modelingUnit': modelingUnit,
        'bpeVocab': bpeVocab,
      };

  final OnlineTransducerModelConfig transducer;
  final OnlineParaformerModelConfig paraformer;
  final OnlineZipformer2CtcModelConfig zipformer2Ctc;
  final OnlineNemoCtcModelConfig nemoCtc;
  final OnlineToneCtcModelConfig toneCtc;

  final String tokens;

  final int numThreads;

  final String provider;

  final bool debug;

  final String modelType;

  final String modelingUnit;

  final String bpeVocab;
}

/// FST decoder settings for CTC-based streaming recognition.
class OnlineCtcFstDecoderConfig {
  const OnlineCtcFstDecoderConfig({this.graph = '', this.maxActive = 3000});

  factory OnlineCtcFstDecoderConfig.fromJson(Map<String, dynamic> json) {
    return OnlineCtcFstDecoderConfig(
      graph: json['graph'] as String? ?? '',
      maxActive: json['maxActive'] as int? ?? 3000,
    );
  }

  @override
  String toString() {
    return 'OnlineCtcFstDecoderConfig(graph: $graph, maxActive: $maxActive)';
  }

  Map<String, dynamic> toJson() => {
        'graph': graph,
        'maxActive': maxActive,
      };

  final String graph;
  final int maxActive;
}

/// Top-level configuration for [OnlineRecognizer].
///
/// This combines feature extraction, the selected online model family,
/// endpointing rules, hotwords, grammar resources, and optional homophone
/// replacement resources.
class OnlineRecognizerConfig {
  const OnlineRecognizerConfig({
    this.feat = const FeatureConfig(),
    required this.model,
    this.decodingMethod = 'greedy_search',
    this.maxActivePaths = 4,
    this.enableEndpoint = true,
    this.rule1MinTrailingSilence = 2.4,
    this.rule2MinTrailingSilence = 1.2,
    this.rule3MinUtteranceLength = 20,
    this.hotwordsFile = '',
    this.hotwordsScore = 1.5,
    this.ctcFstDecoderConfig = const OnlineCtcFstDecoderConfig(),
    this.ruleFsts = '',
    this.ruleFars = '',
    this.blankPenalty = 0.0,
    this.hr = const HomophoneReplacerConfig(),
  });

  factory OnlineRecognizerConfig.fromJson(Map<String, dynamic> json) {
    return OnlineRecognizerConfig(
      feat: FeatureConfig.fromJson(
          json['feat'] as Map<String, dynamic>? ?? const {}),
      model: OnlineModelConfig.fromJson(json['model'] as Map<String, dynamic>),
      decodingMethod: json['decodingMethod'] as String? ?? 'greedy_search',
      maxActivePaths: json['maxActivePaths'] as int? ?? 4,
      enableEndpoint: json['enableEndpoint'] as bool? ?? true,
      rule1MinTrailingSilence:
          (json['rule1MinTrailingSilence'] as num?)?.toDouble() ?? 2.4,
      rule2MinTrailingSilence:
          (json['rule2MinTrailingSilence'] as num?)?.toDouble() ?? 1.2,
      rule3MinUtteranceLength:
          (json['rule3MinUtteranceLength'] as num?)?.toDouble() ?? 20.0,
      hotwordsFile: json['hotwordsFile'] as String? ?? '',
      hotwordsScore: (json['hotwordsScore'] as num?)?.toDouble() ?? 1.5,
      ctcFstDecoderConfig: OnlineCtcFstDecoderConfig.fromJson(
          json['ctcFstDecoderConfig'] as Map<String, dynamic>? ?? const {}),
      ruleFsts: json['ruleFsts'] as String? ?? '',
      ruleFars: json['ruleFars'] as String? ?? '',
      blankPenalty: (json['blankPenalty'] as num?)?.toDouble() ?? 0.0,
      hr: HomophoneReplacerConfig.fromJson(
          json['hr'] as Map<String, dynamic>? ?? const {}),
    );
  }

  @override
  String toString() {
    return 'OnlineRecognizerConfig(feat: $feat, model: $model, decodingMethod: $decodingMethod, maxActivePaths: $maxActivePaths, enableEndpoint: $enableEndpoint, rule1MinTrailingSilence: $rule1MinTrailingSilence, rule2MinTrailingSilence: $rule2MinTrailingSilence, rule3MinUtteranceLength: $rule3MinUtteranceLength, hotwordsFile: $hotwordsFile, hotwordsScore: $hotwordsScore, ctcFstDecoderConfig: $ctcFstDecoderConfig, ruleFsts: $ruleFsts, ruleFars: $ruleFars, blankPenalty: $blankPenalty, hr: $hr)';
  }

  Map<String, dynamic> toJson() => {
        'feat': feat.toJson(),
        'model': model.toJson(),
        'decodingMethod': decodingMethod,
        'maxActivePaths': maxActivePaths,
        'enableEndpoint': enableEndpoint,
        'rule1MinTrailingSilence': rule1MinTrailingSilence,
        'rule2MinTrailingSilence': rule2MinTrailingSilence,
        'rule3MinUtteranceLength': rule3MinUtteranceLength,
        'hotwordsFile': hotwordsFile,
        'hotwordsScore': hotwordsScore,
        'ctcFstDecoderConfig': ctcFstDecoderConfig.toJson(),
        'ruleFsts': ruleFsts,
        'ruleFars': ruleFars,
        'blankPenalty': blankPenalty,
        'hr': hr.toJson(),
      };

  final FeatureConfig feat;
  final OnlineModelConfig model;
  final String decodingMethod;

  final int maxActivePaths;

  final bool enableEndpoint;

  final double rule1MinTrailingSilence;

  final double rule2MinTrailingSilence;

  final double rule3MinUtteranceLength;

  final String hotwordsFile;

  final double hotwordsScore;

  final OnlineCtcFstDecoderConfig ctcFstDecoderConfig;
  final String ruleFsts;
  final String ruleFars;

  final double blankPenalty;
  final HomophoneReplacerConfig hr;
}

/// Streaming recognition result returned by [OnlineRecognizer.getResult].
class OnlineRecognizerResult {
  OnlineRecognizerResult(
      {required this.text, required this.tokens, required this.timestamps});

  factory OnlineRecognizerResult.fromJson(Map<String, dynamic> json) {
    return OnlineRecognizerResult(
      text: json['text'] as String,
      tokens: List<String>.from(json['tokens'] as List),
      timestamps: (json['timestamps'] as List)
          .map<double>((e) => (e as num).toDouble())
          .toList(),
    );
  }

  @override
  String toString() {
    return 'OnlineRecognizerResult(text: $text, tokens: $tokens, timestamps: $timestamps)';
  }

  Map<String, dynamic> toJson() => {
        'text': text,
        'tokens': tokens,
        'timestamps': timestamps,
      };

  final String text;
  final List<String> tokens;
  final List<double> timestamps;
}
