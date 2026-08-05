// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for keyword spotting -- no FFI, works on all platforms.

import './feature_config.dart';
import './online_recognizer_config.dart';

/// Streaming keyword spotting.
///
/// See `dart-api-examples/keyword-spotter/` for end-to-end usage.
///
/// Example:
///
/// ```dart
/// final spotter = KeywordSpotter(
///   KeywordSpotterConfig(
///     model: onlineModelConfig,
///     keywordsFile: './keywords.txt',
///   ),
/// );
///
/// final stream = spotter.createStream();
/// stream.acceptWaveform(samples: chunk, sampleRate: 16000);
/// while (spotter.isReady(stream)) {
///   spotter.decode(stream);
/// }
/// print(spotter.getResult(stream).keyword);
/// ```
class KeywordSpotterConfig {
  const KeywordSpotterConfig({
    this.feat = const FeatureConfig(),
    required this.model,
    this.maxActivePaths = 4,
    this.numTrailingBlanks = 1,
    this.keywordsScore = 1.0,
    this.keywordsThreshold = 0.25,
    this.keywordsFile = '',
    this.keywordsBuf = '',
    this.keywordsBufSize = 0,
  });

  factory KeywordSpotterConfig.fromJson(Map<String, dynamic> json) {
    return KeywordSpotterConfig(
      feat: json['feat'] != null
          ? FeatureConfig.fromJson(json['feat'] as Map<String, dynamic>)
          : const FeatureConfig(),
      model: OnlineModelConfig.fromJson(json['model'] as Map<String, dynamic>),
      maxActivePaths: json['maxActivePaths'] as int? ?? 4,
      numTrailingBlanks: json['numTrailingBlanks'] as int? ?? 1,
      keywordsScore: (json['keywordsScore'] as num?)?.toDouble() ?? 1.0,
      keywordsThreshold:
          (json['keywordsThreshold'] as num?)?.toDouble() ?? 0.25,
      keywordsFile: json['keywordsFile'] as String? ?? '',
      keywordsBuf: json['keywordsBuf'] as String? ?? '',
      keywordsBufSize: json['keywordsBufSize'] as int? ?? 0,
    );
  }

  @override
  String toString() {
    return 'KeywordSpotterConfig(feat: $feat, model: $model, maxActivePaths: $maxActivePaths, numTrailingBlanks: $numTrailingBlanks, keywordsScore: $keywordsScore, keywordsThreshold: $keywordsThreshold, keywordsFile: $keywordsFile, keywordsBuf: $keywordsBuf, keywordsBufSize: $keywordsBufSize)';
  }

  Map<String, dynamic> toJson() => {
        'feat': feat.toJson(),
        'model': model.toJson(),
        'maxActivePaths': maxActivePaths,
        'numTrailingBlanks': numTrailingBlanks,
        'keywordsScore': keywordsScore,
        'keywordsThreshold': keywordsThreshold,
        'keywordsFile': keywordsFile,
        'keywordsBuf': keywordsBuf,
        'keywordsBufSize': keywordsBufSize,
      };

  final FeatureConfig feat;
  final OnlineModelConfig model;

  final int maxActivePaths;
  final int numTrailingBlanks;

  final double keywordsScore;
  final double keywordsThreshold;
  final String keywordsFile;
  final String keywordsBuf;
  final int keywordsBufSize;
}

/// Result returned by [KeywordSpotter.getResult].
class KeywordResult {
  KeywordResult({required this.keyword});

  factory KeywordResult.fromJson(Map<String, dynamic> json) {
    return KeywordResult(
      keyword: json['keyword'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'KeywordResult(keyword: $keyword)';
  }

  Map<String, dynamic> toJson() => {
        'keyword': keyword,
      };

  final String keyword;
}
