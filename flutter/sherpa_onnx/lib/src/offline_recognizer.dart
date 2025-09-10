// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './feature_config.dart';
import './homophone_replacer_config.dart';
import './offline_stream.dart';
import './sherpa_onnx_bindings.dart';
import './utils.dart';

class OfflineTransducerModelConfig {
  const OfflineTransducerModelConfig({
    this.encoder = '',
    this.decoder = '',
    this.joiner = '',
  });

  factory OfflineTransducerModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTransducerModelConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      joiner: json['joiner'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineTransducerModelConfig(encoder: $encoder, decoder: $decoder, joiner: $joiner)';
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

class OfflineParaformerModelConfig {
  const OfflineParaformerModelConfig({this.model = ''});

  factory OfflineParaformerModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineParaformerModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineParaformerModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

class OfflineNemoEncDecCtcModelConfig {
  const OfflineNemoEncDecCtcModelConfig({this.model = ''});

  factory OfflineNemoEncDecCtcModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineNemoEncDecCtcModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineNemoEncDecCtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

class OfflineDolphinModelConfig {
  const OfflineDolphinModelConfig({this.model = ''});

  factory OfflineDolphinModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineDolphinModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineDolphinModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

class OfflineZipformerCtcModelConfig {
  const OfflineZipformerCtcModelConfig({this.model = ''});

  factory OfflineZipformerCtcModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineZipformerCtcModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineZipformerCtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

class OfflineWenetCtcModelConfig {
  const OfflineWenetCtcModelConfig({this.model = ''});

  factory OfflineWenetCtcModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineWenetCtcModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineWenetCtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

class OfflineWhisperModelConfig {
  const OfflineWhisperModelConfig(
      {this.encoder = '',
      this.decoder = '',
      this.language = '',
      this.task = '',
      this.tailPaddings = -1});

  factory OfflineWhisperModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineWhisperModelConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      language: json['language'] as String? ?? '',
      task: json['task'] as String? ?? '',
      tailPaddings: json['tailPaddings'] as int? ?? -1,
    );
  }

  @override
  String toString() {
    return 'OfflineWhisperModelConfig(encoder: $encoder, decoder: $decoder, language: $language, task: $task, tailPaddings: $tailPaddings)';
  }

  Map<String, dynamic> toJson() => {
        'encoder': encoder,
        'decoder': decoder,
        'language': language,
        'task': task,
        'tailPaddings': tailPaddings,
      };

  final String encoder;
  final String decoder;
  final String language;
  final String task;
  final int tailPaddings;
}

class OfflineCanaryModelConfig {
  const OfflineCanaryModelConfig(
      {this.encoder = '',
      this.decoder = '',
      this.srcLang = 'en',
      this.tgtLang = 'en',
      this.usePnc = true});

  factory OfflineCanaryModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineCanaryModelConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      srcLang: json['srcLang'] as String? ?? 'en',
      tgtLang: json['tgtLang'] as String? ?? 'en',
      usePnc: json['usePnc'] as bool? ?? true,
    );
  }

  @override
  String toString() {
    return 'OfflineCanaryModelConfig(encoder: $encoder, decoder: $decoder, srcLang: $srcLang, tgtLang: $tgtLang, usePnc: $usePnc)';
  }

  Map<String, dynamic> toJson() => {
        'encoder': encoder,
        'decoder': decoder,
        'srcLang': srcLang,
        'tgtLang': tgtLang,
        'usePnc': usePnc,
      };

  final String encoder;
  final String decoder;
  final String srcLang;
  final String tgtLang;
  final bool usePnc;
}

class OfflineFireRedAsrModelConfig {
  const OfflineFireRedAsrModelConfig({this.encoder = '', this.decoder = ''});

  factory OfflineFireRedAsrModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineFireRedAsrModelConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineFireRedAsrModelConfig(encoder: $encoder, decoder: $decoder)';
  }

  Map<String, dynamic> toJson() => {
        'encoder': encoder,
        'decoder': decoder,
      };

  final String encoder;
  final String decoder;
}

class OfflineMoonshineModelConfig {
  const OfflineMoonshineModelConfig(
      {this.preprocessor = '',
      this.encoder = '',
      this.uncachedDecoder = '',
      this.cachedDecoder = ''});

  factory OfflineMoonshineModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineMoonshineModelConfig(
      preprocessor: json['preprocessor'] as String? ?? '',
      encoder: json['encoder'] as String? ?? '',
      uncachedDecoder: json['uncachedDecoder'] as String? ?? '',
      cachedDecoder: json['cachedDecoder'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineMoonshineModelConfig(preprocessor: $preprocessor, encoder: $encoder, uncachedDecoder: $uncachedDecoder, cachedDecoder: $cachedDecoder)';
  }

  Map<String, dynamic> toJson() => {
        'preprocessor': preprocessor,
        'encoder': encoder,
        'uncachedDecoder': uncachedDecoder,
        'cachedDecoder': cachedDecoder,
      };

  final String preprocessor;
  final String encoder;
  final String uncachedDecoder;
  final String cachedDecoder;
}

class OfflineTdnnModelConfig {
  const OfflineTdnnModelConfig({this.model = ''});

  factory OfflineTdnnModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTdnnModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineTdnnModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

class OfflineSenseVoiceModelConfig {
  const OfflineSenseVoiceModelConfig({
    this.model = '',
    this.language = '',
    this.useInverseTextNormalization = false,
  });

  factory OfflineSenseVoiceModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineSenseVoiceModelConfig(
      model: json['model'] as String? ?? '',
      language: json['language'] as String? ?? '',
      useInverseTextNormalization:
          json['useInverseTextNormalization'] as bool? ?? false,
    );
  }

  @override
  String toString() {
    return 'OfflineSenseVoiceModelConfig(model: $model, language: $language, useInverseTextNormalization: $useInverseTextNormalization)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'language': language,
        'useInverseTextNormalization': useInverseTextNormalization,
      };

  final String model;
  final String language;
  final bool useInverseTextNormalization;
}

class OfflineLMConfig {
  const OfflineLMConfig({this.model = '', this.scale = 1.0});

  factory OfflineLMConfig.fromJson(Map<String, dynamic> json) {
    return OfflineLMConfig(
      model: json['model'] as String? ?? '',
      scale: (json['scale'] as num?)?.toDouble() ?? 1.0,
    );
  }

  @override
  String toString() {
    return 'OfflineLMConfig(model: $model, scale: $scale)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'scale': scale,
      };

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
    this.senseVoice = const OfflineSenseVoiceModelConfig(),
    this.moonshine = const OfflineMoonshineModelConfig(),
    this.fireRedAsr = const OfflineFireRedAsrModelConfig(),
    this.dolphin = const OfflineDolphinModelConfig(),
    this.zipformerCtc = const OfflineZipformerCtcModelConfig(),
    this.canary = const OfflineCanaryModelConfig(),
    this.wenetCtc = const OfflineWenetCtcModelConfig(),
    required this.tokens,
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
    this.modelType = '',
    this.modelingUnit = '',
    this.bpeVocab = '',
    this.telespeechCtc = '',
  });

  factory OfflineModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineModelConfig(
      transducer: json['transducer'] != null
          ? OfflineTransducerModelConfig.fromJson(
              json['transducer'] as Map<String, dynamic>)
          : const OfflineTransducerModelConfig(),
      paraformer: json['paraformer'] != null
          ? OfflineParaformerModelConfig.fromJson(
              json['paraformer'] as Map<String, dynamic>)
          : const OfflineParaformerModelConfig(),
      nemoCtc: json['nemoCtc'] != null
          ? OfflineNemoEncDecCtcModelConfig.fromJson(
              json['nemoCtc'] as Map<String, dynamic>)
          : const OfflineNemoEncDecCtcModelConfig(),
      whisper: json['whisper'] != null
          ? OfflineWhisperModelConfig.fromJson(
              json['whisper'] as Map<String, dynamic>)
          : const OfflineWhisperModelConfig(),
      tdnn: json['tdnn'] != null
          ? OfflineTdnnModelConfig.fromJson(
              json['tdnn'] as Map<String, dynamic>)
          : const OfflineTdnnModelConfig(),
      senseVoice: json['senseVoice'] != null
          ? OfflineSenseVoiceModelConfig.fromJson(
              json['senseVoice'] as Map<String, dynamic>)
          : const OfflineSenseVoiceModelConfig(),
      moonshine: json['moonshine'] != null
          ? OfflineMoonshineModelConfig.fromJson(
              json['moonshine'] as Map<String, dynamic>)
          : const OfflineMoonshineModelConfig(),
      fireRedAsr: json['fireRedAsr'] != null
          ? OfflineFireRedAsrModelConfig.fromJson(
              json['fireRedAsr'] as Map<String, dynamic>)
          : const OfflineFireRedAsrModelConfig(),
      dolphin: json['dolphin'] != null
          ? OfflineDolphinModelConfig.fromJson(
              json['dolphin'] as Map<String, dynamic>)
          : const OfflineDolphinModelConfig(),
      zipformerCtc: json['zipformerCtc'] != null
          ? OfflineZipformerCtcModelConfig.fromJson(
              json['zipformerCtc'] as Map<String, dynamic>)
          : const OfflineZipformerCtcModelConfig(),
      canary: json['canary'] != null
          ? OfflineCanaryModelConfig.fromJson(
              json['canary'] as Map<String, dynamic>)
          : const OfflineCanaryModelConfig(),
      wenetCtc: json['wenetCtc'] != null
          ? OfflineWenetCtcModelConfig.fromJson(
              json['wenetCtc'] as Map<String, dynamic>)
          : const OfflineWenetCtcModelConfig(),
      tokens: json['tokens'] as String,
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
      modelType: json['modelType'] as String? ?? '',
      modelingUnit: json['modelingUnit'] as String? ?? '',
      bpeVocab: json['bpeVocab'] as String? ?? '',
      telespeechCtc: json['telespeechCtc'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineModelConfig(transducer: $transducer, paraformer: $paraformer, nemoCtc: $nemoCtc, whisper: $whisper, tdnn: $tdnn, senseVoice: $senseVoice, moonshine: $moonshine, fireRedAsr: $fireRedAsr, dolphin: $dolphin, zipformerCtc: $zipformerCtc, canary: $canary, wenetCtc: $wenetCtc, tokens: $tokens, numThreads: $numThreads, debug: $debug, provider: $provider, modelType: $modelType, modelingUnit: $modelingUnit, bpeVocab: $bpeVocab, telespeechCtc: $telespeechCtc)';
  }

  Map<String, dynamic> toJson() => {
        'transducer': transducer.toJson(),
        'paraformer': paraformer.toJson(),
        'nemoCtc': nemoCtc.toJson(),
        'whisper': whisper.toJson(),
        'tdnn': tdnn.toJson(),
        'senseVoice': senseVoice.toJson(),
        'moonshine': moonshine.toJson(),
        'fireRedAsr': fireRedAsr.toJson(),
        'dolphin': dolphin.toJson(),
        'zipformerCtc': zipformerCtc.toJson(),
        'canary': canary.toJson(),
        'wenetCtc': wenetCtc.toJson(),
        'tokens': tokens,
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
        'modelType': modelType,
        'modelingUnit': modelingUnit,
        'bpeVocab': bpeVocab,
        'telespeechCtc': telespeechCtc,
      };

  final OfflineTransducerModelConfig transducer;
  final OfflineParaformerModelConfig paraformer;
  final OfflineNemoEncDecCtcModelConfig nemoCtc;
  final OfflineWhisperModelConfig whisper;
  final OfflineTdnnModelConfig tdnn;
  final OfflineSenseVoiceModelConfig senseVoice;
  final OfflineMoonshineModelConfig moonshine;
  final OfflineFireRedAsrModelConfig fireRedAsr;
  final OfflineDolphinModelConfig dolphin;
  final OfflineZipformerCtcModelConfig zipformerCtc;
  final OfflineCanaryModelConfig canary;
  final OfflineWenetCtcModelConfig wenetCtc;

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
    this.blankPenalty = 0.0,
    this.hr = const HomophoneReplacerConfig(),
  });

  factory OfflineRecognizerConfig.fromJson(Map<String, dynamic> json) {
    return OfflineRecognizerConfig(
      feat: json['feat'] != null
          ? FeatureConfig.fromJson(json['feat'] as Map<String, dynamic>)
          : const FeatureConfig(),
      model: OfflineModelConfig.fromJson(json['model'] as Map<String, dynamic>),
      lm: json['lm'] != null
          ? OfflineLMConfig.fromJson(json['lm'] as Map<String, dynamic>)
          : const OfflineLMConfig(),
      decodingMethod: json['decodingMethod'] as String? ?? 'greedy_search',
      maxActivePaths: json['maxActivePaths'] as int? ?? 4,
      hotwordsFile: json['hotwordsFile'] as String? ?? '',
      hotwordsScore: (json['hotwordsScore'] as num?)?.toDouble() ?? 1.5,
      ruleFsts: json['ruleFsts'] as String? ?? '',
      ruleFars: json['ruleFars'] as String? ?? '',
      blankPenalty: (json['blankPenalty'] as num?)?.toDouble() ?? 0.0,
      hr: HomophoneReplacerConfig.fromJson(json['hr'] as Map<String, dynamic>),
    );
  }

  @override
  String toString() {
    return 'OfflineRecognizerConfig(feat: $feat, model: $model, lm: $lm, decodingMethod: $decodingMethod, maxActivePaths: $maxActivePaths, hotwordsFile: $hotwordsFile, hotwordsScore: $hotwordsScore, ruleFsts: $ruleFsts, ruleFars: $ruleFars, blankPenalty: $blankPenalty, hr: $hr)';
  }

  Map<String, dynamic> toJson() => {
        'feat': feat.toJson(),
        'model': model.toJson(),
        'lm': lm.toJson(),
        'decodingMethod': decodingMethod,
        'maxActivePaths': maxActivePaths,
        'hotwordsFile': hotwordsFile,
        'hotwordsScore': hotwordsScore,
        'ruleFsts': ruleFsts,
        'ruleFars': ruleFars,
        'blankPenalty': blankPenalty,
        'hr': hr.toJson(),
      };

  final FeatureConfig feat;
  final OfflineModelConfig model;
  final OfflineLMConfig lm;
  final String decodingMethod;

  final int maxActivePaths;

  final String hotwordsFile;

  final double hotwordsScore;

  final String ruleFsts;
  final String ruleFars;

  final double blankPenalty;
  final HomophoneReplacerConfig hr;
}

class OfflineRecognizerResult {
  OfflineRecognizerResult(
      {required this.text,
      required this.tokens,
      required this.timestamps,
      required this.lang,
      required this.emotion,
      required this.event});

  factory OfflineRecognizerResult.fromJson(Map<String, dynamic> json) {
    return OfflineRecognizerResult(
      text: json['text'] as String? ?? '',
      tokens: (json['tokens'] as List?)?.map((e) => e as String).toList() ?? [],
      timestamps: (json['timestamps'] as List?)
              ?.map((e) => (e as num).toDouble())
              .toList() ??
          [],
      lang: json['lang'] as String? ?? '',
      emotion: json['emotion'] as String? ?? '',
      event: json['event'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineRecognizerResult(text: $text, tokens: $tokens, timestamps: $timestamps, lang: $lang, emotion: $emotion, event: $event)';
  }

  Map<String, dynamic> toJson() => {
        'text': text,
        'tokens': tokens,
        'timestamps': timestamps,
        'lang': lang,
        'emotion': emotion,
        'event': event,
      };

  final String text;
  final List<String> tokens;
  final List<double> timestamps;
  final String lang;
  final String emotion;
  final String event;
}

class OfflineRecognizer {
  OfflineRecognizer.fromPtr({required this.ptr, required this.config});

  OfflineRecognizer._({required this.ptr, required this.config});

  void free() {
    SherpaOnnxBindings.destroyOfflineRecognizer?.call(ptr);
    ptr = nullptr;
  }

  /// The user is responsible to call the OfflineRecognizer.free()
  /// method of the returned instance to avoid memory leak.

  factory OfflineRecognizer(OfflineRecognizerConfig config) {
    final c = convertConfig(config);

    if (SherpaOnnxBindings.createOfflineRecognizer == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final ptr = SherpaOnnxBindings.createOfflineRecognizer?.call(c) ?? nullptr;

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create offline recognizer. Please check your config");
    }

    freeConfig(c);

    return OfflineRecognizer._(ptr: ptr, config: config);
  }

  void setConfig(OfflineRecognizerConfig config) {
    final c = convertConfig(config);

    SherpaOnnxBindings.offlineRecognizerSetConfig?.call(ptr, c);

    freeConfig(c);
    // we don't update this.config
  }

  static Pointer<SherpaOnnxOfflineRecognizerConfig> convertConfig(
      OfflineRecognizerConfig config) {
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

    c.ref.model.senseVoice.model = config.model.senseVoice.model.toNativeUtf8();

    c.ref.model.senseVoice.language =
        config.model.senseVoice.language.toNativeUtf8();

    c.ref.model.senseVoice.useInverseTextNormalization =
        config.model.senseVoice.useInverseTextNormalization ? 1 : 0;

    c.ref.model.moonshine.preprocessor =
        config.model.moonshine.preprocessor.toNativeUtf8();
    c.ref.model.moonshine.encoder =
        config.model.moonshine.encoder.toNativeUtf8();
    c.ref.model.moonshine.uncachedDecoder =
        config.model.moonshine.uncachedDecoder.toNativeUtf8();
    c.ref.model.moonshine.cachedDecoder =
        config.model.moonshine.cachedDecoder.toNativeUtf8();

    // FireRedAsr
    c.ref.model.fireRedAsr.encoder =
        config.model.fireRedAsr.encoder.toNativeUtf8();
    c.ref.model.fireRedAsr.decoder =
        config.model.fireRedAsr.decoder.toNativeUtf8();

    c.ref.model.dolphin.model = config.model.dolphin.model.toNativeUtf8();
    c.ref.model.zipformerCtc.model =
        config.model.zipformerCtc.model.toNativeUtf8();

    c.ref.model.canary.encoder = config.model.canary.encoder.toNativeUtf8();
    c.ref.model.canary.decoder = config.model.canary.decoder.toNativeUtf8();
    c.ref.model.canary.srcLang = config.model.canary.srcLang.toNativeUtf8();
    c.ref.model.canary.tgtLang = config.model.canary.tgtLang.toNativeUtf8();
    c.ref.model.canary.usePnc = config.model.canary.usePnc ? 1 : 0;

    c.ref.model.wenetCtc.model = config.model.wenetCtc.model.toNativeUtf8();

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

    c.ref.blankPenalty = config.blankPenalty;

    c.ref.hr.dictDir = config.hr.dictDir.toNativeUtf8();
    c.ref.hr.lexicon = config.hr.lexicon.toNativeUtf8();
    c.ref.hr.ruleFsts = config.hr.ruleFsts.toNativeUtf8();

    return c;
  }

  static void freeConfig(Pointer<SherpaOnnxOfflineRecognizerConfig> c) {
    calloc.free(c.ref.hr.dictDir);
    calloc.free(c.ref.hr.lexicon);
    calloc.free(c.ref.hr.ruleFsts);
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
    calloc.free(c.ref.model.wenetCtc.model);
    calloc.free(c.ref.model.canary.tgtLang);
    calloc.free(c.ref.model.canary.srcLang);
    calloc.free(c.ref.model.canary.decoder);
    calloc.free(c.ref.model.canary.encoder);
    calloc.free(c.ref.model.zipformerCtc.model);
    calloc.free(c.ref.model.dolphin.model);
    calloc.free(c.ref.model.fireRedAsr.decoder);
    calloc.free(c.ref.model.fireRedAsr.encoder);
    calloc.free(c.ref.model.moonshine.cachedDecoder);
    calloc.free(c.ref.model.moonshine.uncachedDecoder);
    calloc.free(c.ref.model.moonshine.encoder);
    calloc.free(c.ref.model.moonshine.preprocessor);
    calloc.free(c.ref.model.senseVoice.language);
    calloc.free(c.ref.model.senseVoice.model);
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
      return OfflineRecognizerResult(
          text: '',
          tokens: [],
          timestamps: [],
          lang: '',
          emotion: '',
          event: '');
    }

    final parsedJson = jsonDecode(toDartString(json));

    SherpaOnnxBindings.destroyOfflineStreamResultJson?.call(json);

    return OfflineRecognizerResult(
        text: parsedJson['text'],
        tokens: List<String>.from(parsedJson['tokens']),
        timestamps: List<double>.from(parsedJson['timestamps']),
        lang: parsedJson['lang'],
        emotion: parsedJson['emotion'],
        event: parsedJson['event']);
  }

  Pointer<SherpaOnnxOfflineRecognizer> ptr;
  OfflineRecognizerConfig config;
}
