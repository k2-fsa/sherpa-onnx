// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './feature_config.dart';
import './offline_stream.dart';
import './sherpa_onnx_bindings.dart';
import './utils.dart';

class OfflineTransducerModelConfig {
  const OfflineTransducerModelConfig({
    this.encoder = '',
    this.decoder = '',
    this.joiner = '',
  });

  @override
  String toString() {
    return 'OfflineTransducerModelConfig(encoder: $encoder, decoder: $decoder, joiner: $joiner)';
  }

  final String encoder;
  final String decoder;
  final String joiner;
}

class OfflineParaformerModelConfig {
  const OfflineParaformerModelConfig({this.model = ''});

  @override
  String toString() {
    return 'OfflineParaformerModelConfig(model: $model)';
  }

  final String model;
}

class OfflineNemoEncDecCtcModelConfig {
  const OfflineNemoEncDecCtcModelConfig({this.model = ''});

  @override
  String toString() {
    return 'OfflineNemoEncDecCtcModelConfig(model: $model)';
  }

  final String model;
}

class OfflineWhisperModelConfig {
  const OfflineWhisperModelConfig(
      {this.encoder = '',
      this.decoder = '',
      this.language = '',
      this.task = '',
      this.tailPaddings = -1});

  @override
  String toString() {
    return 'OfflineWhisperModelConfig(encoder: $encoder, decoder: $decoder, language: $language, task: $task, tailPaddings: $tailPaddings)';
  }

  final String encoder;
  final String decoder;
  final String language;
  final String task;
  final int tailPaddings;
}

class OfflineTdnnModelConfig {
  const OfflineTdnnModelConfig({this.model = ''});

  @override
  String toString() {
    return 'OfflineTdnnModelConfig(model: $model)';
  }

  final String model;
}

class OfflineLMConfig {
  const OfflineLMConfig({this.model = '', this.scale = 1.0});

  @override
  String toString() {
    return 'OfflineLMConfig(model: $model, scale: $scale)';
  }

  final String model;
  final double scale;
}

class OfflineModelConfig {
  const OfflineModelConfig({
    this.transducer = const OfflineTransducerModelConfig(),
    this.paraformer = const OfflineParaformerModelConfig(),
    this.nemoCtc = const OfflineNemoEncDecCtcModelConfig(),
    this.whisper = const OfflineWhisperModelConfig(),
    this.tdnn = const OfflineTdnnModelConfig(),
    required this.tokens,
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
    this.modelType = '',
    this.modelingUnit = '',
    this.bpeVocab = '',
    this.telespeechCtc = '',
  });

  @override
  String toString() {
    return 'OfflineModelConfig(transducer: $transducer, paraformer: $paraformer, nemoCtc: $nemoCtc, whisper: $whisper, tdnn: $tdnn, tokens: $tokens, numThreads: $numThreads, debug: $debug, provider: $provider, modelType: $modelType, modelingUnit: $modelingUnit, bpeVocab: $bpeVocab, telespeechCtc: $telespeechCtc)';
  }

  final OfflineTransducerModelConfig transducer;
  final OfflineParaformerModelConfig paraformer;
  final OfflineNemoEncDecCtcModelConfig nemoCtc;
  final OfflineWhisperModelConfig whisper;
  final OfflineTdnnModelConfig tdnn;

  final String tokens;
  final int numThreads;
  final bool debug;
  final String provider;
  final String modelType;
  final String modelingUnit;
  final String bpeVocab;
  final String telespeechCtc;
}

class OfflineRecognizerConfig {
  const OfflineRecognizerConfig({
    this.feat = const FeatureConfig(),
    required this.model,
    this.lm = const OfflineLMConfig(),
    this.decodingMethod = 'greedy_search',
    this.maxActivePaths = 4,
    this.hotwordsFile = '',
    this.hotwordsScore = 1.5,
    this.ruleFsts = '',
    this.ruleFars = '',
  });

  @override
  String toString() {
    return 'OfflineRecognizerConfig(feat: $feat, model: $model, lm: $lm, decodingMethod: $decodingMethod, maxActivePaths: $maxActivePaths, hotwordsFile: $hotwordsFile, hotwordsScore: $hotwordsScore, ruleFsts: $ruleFsts, ruleFars: $ruleFars)';
  }

  final FeatureConfig feat;
  final OfflineModelConfig model;
  final OfflineLMConfig lm;
  final String decodingMethod;

  final int maxActivePaths;

  final String hotwordsFile;

  final double hotwordsScore;

  final String ruleFsts;
  final String ruleFars;
}

class OfflineRecognizerResult {
  OfflineRecognizerResult(
      {required this.text, required this.tokens, required this.timestamps});

  @override
  String toString() {
    return 'OfflineRecognizerResult(text: $text, tokens: $tokens, timestamps: $timestamps)';
  }

  final String text;
  final List<String> tokens;
  final List<double> timestamps;
}

class OfflineRecognizer {
  OfflineRecognizer._({required this.ptr, required this.config});

  void free() {
    SherpaOnnxBindings.destroyOfflineRecognizer?.call(ptr);
    ptr = nullptr;
  }

  /// The user is responsible to call the OfflineRecognizer.free()
  /// method of the returned instance to avoid memory leak.
  factory OfflineRecognizer(OfflineRecognizerConfig config) {
    final c = calloc<SherpaOnnxOfflineRecognizerConfig>();

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
    c.ref.model.paraformer.model = config.model.paraformer.model.toNativeUtf8();

    // nemoCtc
    c.ref.model.nemoCtc.model = config.model.nemoCtc.model.toNativeUtf8();

    // whisper
    c.ref.model.whisper.encoder = config.model.whisper.encoder.toNativeUtf8();

    c.ref.model.whisper.decoder = config.model.whisper.decoder.toNativeUtf8();

    c.ref.model.whisper.language = config.model.whisper.language.toNativeUtf8();

    c.ref.model.whisper.task = config.model.whisper.task.toNativeUtf8();

    c.ref.model.whisper.tailPaddings = config.model.whisper.tailPaddings;

    c.ref.model.tdnn.model = config.model.tdnn.model.toNativeUtf8();

    c.ref.model.tokens = config.model.tokens.toNativeUtf8();

    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.debug = config.model.debug ? 1 : 0;
    c.ref.model.provider = config.model.provider.toNativeUtf8();
    c.ref.model.modelType = config.model.modelType.toNativeUtf8();
    c.ref.model.modelingUnit = config.model.modelingUnit.toNativeUtf8();
    c.ref.model.bpeVocab = config.model.bpeVocab.toNativeUtf8();
    c.ref.model.telespeechCtc = config.model.telespeechCtc.toNativeUtf8();

    c.ref.lm.model = config.lm.model.toNativeUtf8();
    c.ref.lm.scale = config.lm.scale;

    c.ref.decodingMethod = config.decodingMethod.toNativeUtf8();
    c.ref.maxActivePaths = config.maxActivePaths;

    c.ref.hotwordsFile = config.hotwordsFile.toNativeUtf8();
    c.ref.hotwordsScore = config.hotwordsScore;

    c.ref.ruleFsts = config.ruleFsts.toNativeUtf8();
    c.ref.ruleFars = config.ruleFars.toNativeUtf8();

    final ptr = SherpaOnnxBindings.createOfflineRecognizer?.call(c) ?? nullptr;

    calloc.free(c.ref.ruleFars);
    calloc.free(c.ref.ruleFsts);
    calloc.free(c.ref.hotwordsFile);
    calloc.free(c.ref.decodingMethod);
    calloc.free(c.ref.lm.model);
    calloc.free(c.ref.model.telespeechCtc);
    calloc.free(c.ref.model.bpeVocab);
    calloc.free(c.ref.model.modelingUnit);
    calloc.free(c.ref.model.modelType);
    calloc.free(c.ref.model.provider);
    calloc.free(c.ref.model.tokens);
    calloc.free(c.ref.model.tdnn.model);
    calloc.free(c.ref.model.whisper.task);
    calloc.free(c.ref.model.whisper.language);
    calloc.free(c.ref.model.whisper.decoder);
    calloc.free(c.ref.model.whisper.encoder);
    calloc.free(c.ref.model.nemoCtc.model);
    calloc.free(c.ref.model.paraformer.model);
    calloc.free(c.ref.model.transducer.encoder);
    calloc.free(c.ref.model.transducer.decoder);
    calloc.free(c.ref.model.transducer.joiner);
    calloc.free(c);

    return OfflineRecognizer._(ptr: ptr, config: config);
  }

  /// The user has to invoke stream.free() on the returned instance
  /// to avoid memory leak
  OfflineStream createStream() {
    final p = SherpaOnnxBindings.createOfflineStream?.call(ptr) ?? nullptr;
    return OfflineStream(ptr: p);
  }

  void decode(OfflineStream stream) {
    SherpaOnnxBindings.decodeOfflineStream?.call(ptr, stream.ptr);
  }

  OfflineRecognizerResult getResult(OfflineStream stream) {
    final json =
        SherpaOnnxBindings.getOfflineStreamResultAsJson?.call(stream.ptr) ??
            nullptr;
    if (json == nullptr) {
      return OfflineRecognizerResult(text: '', tokens: [], timestamps: []);
    }

    final parsedJson = jsonDecode(toDartString(json));

    SherpaOnnxBindings.destroyOfflineStreamResultJson?.call(json);

    return OfflineRecognizerResult(
        text: parsedJson['text'],
        tokens: List<String>.from(parsedJson['tokens']),
        timestamps: List<double>.from(parsedJson['timestamps']));
  }

  Pointer<SherpaOnnxOfflineRecognizer> ptr;
  OfflineRecognizerConfig config;
}
