// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './feature_config.dart';
import './online_stream.dart';
import './online_recognizer.dart';
import './sherpa_onnx_bindings.dart';
import './utils.dart';

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

/// Streaming keyword spotter.
class KeywordSpotter {
  KeywordSpotter.fromPtr({required this.ptr, required this.config});

  KeywordSpotter._({required this.ptr, required this.config});

  /// Create a keyword spotter from [config].
  factory KeywordSpotter(KeywordSpotterConfig config) {
    final c = calloc<SherpaOnnxKeywordSpotterConfig>();
    c.ref.feat.sampleRate = config.feat.sampleRate;
    c.ref.feat.featureDim = config.feat.featureDim;

    // transducer
    c.ref.model.transducer.encoder =
        config.model.transducer.encoder.toNativeUtf8();
    c.ref.model.transducer.decoder =
        config.model.transducer.decoder.toNativeUtf8();
    c.ref.model.transducer.joiner =
        config.model.transducer.joiner.toNativeUtf8();

    // paraformer
    c.ref.model.paraformer.encoder =
        config.model.paraformer.encoder.toNativeUtf8();
    c.ref.model.paraformer.decoder =
        config.model.paraformer.decoder.toNativeUtf8();

    // zipformer2Ctc
    c.ref.model.zipformer2Ctc.model =
        config.model.zipformer2Ctc.model.toNativeUtf8();

    // nemoCtc
    c.ref.model.nemoCtc.model = config.model.nemoCtc.model.toNativeUtf8();

    c.ref.model.tokens = config.model.tokens.toNativeUtf8();
    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.provider = config.model.provider.toNativeUtf8();
    c.ref.model.debug = config.model.debug ? 1 : 0;
    c.ref.model.modelType = config.model.modelType.toNativeUtf8();
    c.ref.model.modelingUnit = config.model.modelingUnit.toNativeUtf8();
    c.ref.model.bpeVocab = config.model.bpeVocab.toNativeUtf8();

    c.ref.maxActivePaths = config.maxActivePaths;
    c.ref.numTrailingBlanks = config.numTrailingBlanks;
    c.ref.keywordsScore = config.keywordsScore;
    c.ref.keywordsThreshold = config.keywordsThreshold;
    c.ref.keywordsFile = config.keywordsFile.toNativeUtf8();
    c.ref.keywordsBuf = config.keywordsBuf.toNativeUtf8();
    c.ref.keywordsBufSize = config.keywordsBufSize;

    if (SherpaOnnxBindings.createKeywordSpotter == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final ptr = SherpaOnnxBindings.createKeywordSpotter?.call(c) ?? nullptr;

    calloc.free(c.ref.keywordsBuf);
    calloc.free(c.ref.keywordsFile);
    calloc.free(c.ref.model.bpeVocab);
    calloc.free(c.ref.model.modelingUnit);
    calloc.free(c.ref.model.modelType);
    calloc.free(c.ref.model.provider);
    calloc.free(c.ref.model.tokens);
    calloc.free(c.ref.model.nemoCtc.model);
    calloc.free(c.ref.model.zipformer2Ctc.model);
    calloc.free(c.ref.model.paraformer.encoder);
    calloc.free(c.ref.model.paraformer.decoder);

    calloc.free(c.ref.model.transducer.encoder);
    calloc.free(c.ref.model.transducer.decoder);
    calloc.free(c.ref.model.transducer.joiner);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception("Failed to create kws. Please check your config");
    }

    return KeywordSpotter._(ptr: ptr, config: config);
  }

  /// Release the native keyword spotter.
  void free() {
    if (SherpaOnnxBindings.destroyKeywordSpotter == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroyKeywordSpotter?.call(ptr);
    ptr = nullptr;
  }

  /// Create a streaming input stream.
  ///
  /// If [keywords] is provided, it overrides the configured keywords for that
  /// stream.
  OnlineStream createStream({String keywords = ''}) {
    if (keywords == '') {
      if (SherpaOnnxBindings.createKeywordStream == null) {
        throw Exception("Please initialize sherpa-onnx first");
      }
    } else {
      if (SherpaOnnxBindings.createKeywordStreamWithKeywords == null) {
        throw Exception("Please initialize sherpa-onnx first");
      }
    }

    if (ptr == nullptr) {
      throw Exception("Failed to create online stream");
    }

    if (keywords == '') {
      final p = SherpaOnnxBindings.createKeywordStream?.call(ptr) ?? nullptr;
      if (p == nullptr) {
        throw Exception("Failed to create online stream");
      }
      return OnlineStream(ptr: p);
    }

    final utf8 = keywords.toNativeUtf8();
    final p =
        SherpaOnnxBindings.createKeywordStreamWithKeywords?.call(ptr, utf8) ??
            nullptr;
    calloc.free(utf8);

    if (p == nullptr) {
      throw Exception("Failed to create online stream");
    }

    return OnlineStream(ptr: p);
  }

  /// Return `true` if [stream] has enough audio for another decode step.
  bool isReady(OnlineStream stream) {
    if (SherpaOnnxBindings.isKeywordStreamReady == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return false;
    }

    int ready =
        SherpaOnnxBindings.isKeywordStreamReady?.call(ptr, stream.ptr) ?? 0;

    return ready == 1;
  }

  /// Fetch the current keyword spotting result for [stream].
  KeywordResult getResult(OnlineStream stream) {
    if (SherpaOnnxBindings.getKeywordResultAsJson == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return KeywordResult(keyword: '');
    }

    final json =
        SherpaOnnxBindings.getKeywordResultAsJson?.call(ptr, stream.ptr) ??
            nullptr;
    if (json == nullptr) {
      return KeywordResult(keyword: '');
    }

    final parsedJson = jsonDecode(toDartString(json));

    SherpaOnnxBindings.freeKeywordResultJson?.call(json);

    return KeywordResult(
      keyword: parsedJson['keyword'],
    );
  }

  /// Decode one incremental step for [stream].
  void decode(OnlineStream stream) {
    if (SherpaOnnxBindings.decodeKeywordStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.decodeKeywordStream?.call(ptr, stream.ptr);
  }

  /// Reset the internal state for [stream].
  void reset(OnlineStream stream) {
    if (SherpaOnnxBindings.resetKeywordStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.resetKeywordStream?.call(ptr, stream.ptr);
  }

  Pointer<SherpaOnnxKeywordSpotter> ptr;
  KeywordSpotterConfig config;
}
