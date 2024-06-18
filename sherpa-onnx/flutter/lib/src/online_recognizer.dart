// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './feature_config.dart';
import './online_stream.dart';
import './sherpa_onnx_bindings.dart';

class OnlineTransducerModelConfig {
  const OnlineTransducerModelConfig({
    this.encoder = '',
    this.decoder = '',
    this.joiner = '',
  });

  @override
  String toString() {
    return 'OnlineTransducerModelConfig(encoder: $encoder, decoder: $decoder, joiner: $joiner)';
  }

  final String encoder;
  final String decoder;
  final String joiner;
}

class OnlineParaformerModelConfig {
  const OnlineParaformerModelConfig({this.encoder = '', this.decoder = ''});

  @override
  String toString() {
    return 'OnlineParaformerModelConfig(encoder: $encoder, decoder: $decoder)';
  }

  final String encoder;
  final String decoder;
}

class OnlineZipformer2CtcModelConfig {
  const OnlineZipformer2CtcModelConfig({this.model = ''});

  @override
  String toString() {
    return 'OnlineZipformer2CtcModelConfig(model: $model)';
  }

  final String model;
}

class OnlineModelConfig {
  const OnlineModelConfig({
    this.transducer = const OnlineTransducerModelConfig(),
    this.paraformer = const OnlineParaformerModelConfig(),
    this.zipformer2Ctc = const OnlineZipformer2CtcModelConfig(),
    required this.tokens,
    this.numThreads = 1,
    this.provider = 'cpu',
    this.debug = true,
    this.modelType = '',
    this.modelingUnit = '',
    this.bpeVocab = '',
  });

  @override
  String toString() {
    return 'OnlineModelConfig(transducer: $transducer, paraformer: $paraformer, zipformer2Ctc: $zipformer2Ctc, tokens: $tokens, numThreads: $numThreads, provider: $provider, debug: $debug, modelType: $modelType, modelingUnit: $modelingUnit, bpeVocab: $bpeVocab)';
  }

  final OnlineTransducerModelConfig transducer;
  final OnlineParaformerModelConfig paraformer;
  final OnlineZipformer2CtcModelConfig zipformer2Ctc;

  final String tokens;

  final int numThreads;

  final String provider;

  final bool debug;

  final String modelType;

  final String modelingUnit;

  final String bpeVocab;
}

class OnlineCtcFstDecoderConfig {
  const OnlineCtcFstDecoderConfig({this.graph = '', this.maxActive = 3000});

  @override
  String toString() {
    return 'OnlineCtcFstDecoderConfig(graph: $graph, maxActive: $maxActive)';
  }

  final String graph;
  final int maxActive;
}

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
  });

  @override
  String toString() {
    return 'OnlineRecognizerConfig(feat: $feat, model: $model, decodingMethod: $decodingMethod, maxActivePaths: $maxActivePaths, enableEndpoint: $enableEndpoint, rule1MinTrailingSilence: $rule1MinTrailingSilence, rule2MinTrailingSilence: $rule2MinTrailingSilence, rule3MinUtteranceLength: $rule3MinUtteranceLength, hotwordsFile: $hotwordsFile, hotwordsScore: $hotwordsScore, ctcFstDecoderConfig: $ctcFstDecoderConfig, ruleFsts: $ruleFsts, ruleFars: $ruleFars)';
  }

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
}

class OnlineRecognizerResult {
  OnlineRecognizerResult(
      {required this.text, required this.tokens, required this.timestamps});

  @override
  String toString() {
    return 'OnlineRecognizerResult(text: $text, tokens: $tokens, timestamps: $timestamps)';
  }

  final String text;
  final List<String> tokens;
  final List<double> timestamps;
}

class OnlineRecognizer {
  OnlineRecognizer._({required this.ptr, required this.config});

  /// The user is responsible to call the OnlineRecognizer.free()
  /// method of the returned instance to avoid memory leak.
  factory OnlineRecognizer(OnlineRecognizerConfig config) {
    final c = calloc<SherpaOnnxOnlineRecognizerConfig>();
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

    c.ref.model.tokens = config.model.tokens.toNativeUtf8();
    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.provider = config.model.provider.toNativeUtf8();
    c.ref.model.debug = config.model.debug ? 1 : 0;
    c.ref.model.modelType = config.model.modelType.toNativeUtf8();
    c.ref.model.modelingUnit = config.model.modelingUnit.toNativeUtf8();
    c.ref.model.bpeVocab = config.model.bpeVocab.toNativeUtf8();

    c.ref.decodingMethod = config.decodingMethod.toNativeUtf8();
    c.ref.maxActivePaths = config.maxActivePaths;
    c.ref.enableEndpoint = config.enableEndpoint ? 1 : 0;
    c.ref.rule1MinTrailingSilence = config.rule1MinTrailingSilence;
    c.ref.rule2MinTrailingSilence = config.rule2MinTrailingSilence;
    c.ref.rule3MinUtteranceLength = config.rule3MinUtteranceLength;
    c.ref.hotwordsFile = config.hotwordsFile.toNativeUtf8();
    c.ref.hotwordsScore = config.hotwordsScore;

    c.ref.ctcFstDecoderConfig.graph =
        config.ctcFstDecoderConfig.graph.toNativeUtf8();
    c.ref.ctcFstDecoderConfig.maxActive = config.ctcFstDecoderConfig.maxActive;
    c.ref.ruleFsts = config.ruleFsts.toNativeUtf8();
    c.ref.ruleFars = config.ruleFars.toNativeUtf8();

    final ptr = SherpaOnnxBindings.createOnlineRecognizer?.call(c) ?? nullptr;

    calloc.free(c.ref.ruleFars);
    calloc.free(c.ref.ruleFsts);
    calloc.free(c.ref.ctcFstDecoderConfig.graph);
    calloc.free(c.ref.hotwordsFile);
    calloc.free(c.ref.decodingMethod);
    calloc.free(c.ref.model.bpeVocab);
    calloc.free(c.ref.model.modelingUnit);
    calloc.free(c.ref.model.modelType);
    calloc.free(c.ref.model.provider);
    calloc.free(c.ref.model.tokens);
    calloc.free(c.ref.model.zipformer2Ctc.model);
    calloc.free(c.ref.model.paraformer.encoder);
    calloc.free(c.ref.model.paraformer.decoder);

    calloc.free(c.ref.model.transducer.encoder);
    calloc.free(c.ref.model.transducer.decoder);
    calloc.free(c.ref.model.transducer.joiner);
    calloc.free(c);

    return OnlineRecognizer._(ptr: ptr, config: config);
  }

  void free() {
    SherpaOnnxBindings.destroyOnlineRecognizer?.call(ptr);
    ptr = nullptr;
  }

  /// The user has to invoke stream.free() on the returned instance
  /// to avoid memory leak
  OnlineStream createStream({String hotwords = ''}) {
    if (hotwords == '') {
      final p = SherpaOnnxBindings.createOnlineStream?.call(ptr) ?? nullptr;
      return OnlineStream(ptr: p);
    }

    final utf8 = hotwords.toNativeUtf8();
    final p =
        SherpaOnnxBindings.createOnlineStreamWithHotwords?.call(ptr, utf8) ??
            nullptr;
    calloc.free(utf8);
    return OnlineStream(ptr: p);
  }

  bool isReady(OnlineStream stream) {
    int ready =
        SherpaOnnxBindings.isOnlineStreamReady?.call(ptr, stream.ptr) ?? 0;

    return ready == 1;
  }

  OnlineRecognizerResult getResult(OnlineStream stream) {
    final json =
        SherpaOnnxBindings.getOnlineStreamResultAsJson?.call(ptr, stream.ptr) ??
            nullptr;
    if (json == nullptr) {
      return OnlineRecognizerResult(text: '', tokens: [], timestamps: []);
    }

    final parsedJson = jsonDecode(json.toDartString());

    SherpaOnnxBindings.destroyOnlineStreamResultJson?.call(json);

    return OnlineRecognizerResult(
        text: parsedJson['text'],
        tokens: List<String>.from(parsedJson['tokens']),
        timestamps: List<double>.from(parsedJson['timestamps']));
  }

  void reset(OnlineStream stream) {
    SherpaOnnxBindings.reset?.call(ptr, stream.ptr);
  }

  void decode(OnlineStream stream) {
    SherpaOnnxBindings.decodeOnlineStream?.call(ptr, stream.ptr);
  }

  bool isEndpoint(OnlineStream stream) {
    int yes = SherpaOnnxBindings.isEndpoint?.call(ptr, stream.ptr) ?? 0;

    return yes == 1;
  }

  Pointer<SherpaOnnxOnlineRecognizer> ptr;
  OnlineRecognizerConfig config;
}
