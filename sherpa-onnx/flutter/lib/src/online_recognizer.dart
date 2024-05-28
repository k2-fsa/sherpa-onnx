// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './online_stream.dart';
import './sherpa_onnx_bindings.dart';

class FeatureConfig {
  const FeatureConfig({this.sampleRate = 16000, this.featureDim = 80});

  @override
  String toString() {
    return 'FeatureConfig(sampleRate: $sampleRate, featureDim: $featureDim)';
  }

  final int sampleRate;
  final int featureDim;
}

class OnlineTransducerModelConfig {
  const OnlineTransducerModelConfig(
      {this.encoder = '', this.decoder = '', this.joiner = ''});

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
  });

  @override
  String toString() {
    return 'OnlineModelConfig(transducer: $transducer, paraformer: $paraformer, zipformer2Ctc: $zipformer2Ctc, tokens: $tokens, numThreads: $numThreads, provider: $provider, debug: $debug, modelType: $modelType)';
  }

  final OnlineTransducerModelConfig transducer;
  final OnlineParaformerModelConfig paraformer;
  final OnlineZipformer2CtcModelConfig zipformer2Ctc;

  final String tokens;

  final int numThreads;

  final String provider;

  final bool debug;

  final String modelType;
}

class OnlineCtcFstDecoderConfig extends Struct {
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
    this.model = const OnlineModelConfig(),
    this.decodingMethod = 'greedy_search',
    this.maxActivePaths = 4,
    this.enableEndpoint = true,
    this.rule1MinTrailingSilence = 2.4,
    this.rule2MinTrailingSilence = 1.2,
    this.rule3MinUtteranceLength = 20,
    this.rule3MinUtteranceLength = 20,
    this.hotwordsFile = '',
    this.hotwordsScore = 1.5,
  });

  @override
  String toString() {
    return 'OnlineRecognizerConfig(feat: $feat, model: $model, decodingMethod: $decodingMethod, maxActivePaths: $maxActivePaths, enableEndpoint: $enableEndpoint, rule1MinTrailingSilence: $rule1MinTrailingSilence, rule2MinTrailingSilence: $rule2MinTrailingSilence, rule3MinUtteranceLength: $rule3MinUtteranceLength, hotwordsFile: $hotwordsFile, hotwordsScore: $hotwordsScore, ctcFstDecoderConfig: $ctcFstDecoderConfig)';
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
}

class OnlineRecognizer {
  OnlineRecognizer._({required this.ptr, required this.config});
  factory OnlineRecognizer(OnlineRecognizerConfig config) {
    final c = calloc<SherpaOnnxOnlineRecognizerConfig>();
    c.ref.feat.sampleRate = config.feat.sampleRate;
    c.ref.feat.featureDim = config.feat.featureDim;
  }

  Pointer<SherpaOnnxOnlineRecognizer> ptr;
  OnlineRecognizerConfig config;
}
