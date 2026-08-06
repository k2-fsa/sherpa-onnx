// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for offline recognition -- no FFI, works on all platforms.

import './feature_config.dart';
import './homophone_replacer_config.dart';

/// Model files for an offline transducer recognizer.
///
/// This family is also used by NeMo Parakeet TDT-style examples.
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

/// Model files for an offline Paraformer recognizer.
class OfflineParaformerModelConfig {
  const OfflineParaformerModelConfig({this.model = ''});

  factory OfflineParaformerModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineParaformerModelConfig(model: json['model'] as String? ?? '');
  }

  @override
  String toString() {
    return 'OfflineParaformerModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {'model': model};

  final String model;
}

/// Model files for an offline NeMo CTC recognizer.
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

  Map<String, dynamic> toJson() => {'model': model};

  final String model;
}

/// Model files for an offline Dolphin recognizer.
class OfflineDolphinModelConfig {
  const OfflineDolphinModelConfig({this.model = ''});

  factory OfflineDolphinModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineDolphinModelConfig(model: json['model'] as String? ?? '');
  }

  @override
  String toString() {
    return 'OfflineDolphinModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {'model': model};

  final String model;
}

/// Model files for an offline Zipformer CTC recognizer.
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

  Map<String, dynamic> toJson() => {'model': model};

  final String model;
}

/// Model files for an offline WeNet CTC recognizer.
class OfflineWenetCtcModelConfig {
  const OfflineWenetCtcModelConfig({this.model = ''});

  factory OfflineWenetCtcModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineWenetCtcModelConfig(model: json['model'] as String? ?? '');
  }

  @override
  String toString() {
    return 'OfflineWenetCtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {'model': model};

  final String model;
}

/// Model files for the omnilingual ASR CTC recognizer.
class OfflineOmnilingualAsrCtcModelConfig {
  const OfflineOmnilingualAsrCtcModelConfig({this.model = ''});

  factory OfflineOmnilingualAsrCtcModelConfig.fromJson(
    Map<String, dynamic> json,
  ) {
    return OfflineOmnilingualAsrCtcModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineOmnilingualAsrCtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {'model': model};

  final String model;
}

/// Model files for the MedASR CTC recognizer.
class OfflineMedAsrCtcModelConfig {
  const OfflineMedAsrCtcModelConfig({this.model = ''});

  factory OfflineMedAsrCtcModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineMedAsrCtcModelConfig(model: json['model'] as String? ?? '');
  }

  @override
  String toString() {
    return 'OfflineMedAsrCtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {'model': model};

  final String model;
}

/// Model files for the Fire-Red-ASR CTC recognizer.
class OfflineFireRedAsrCtcModelConfig {
  const OfflineFireRedAsrCtcModelConfig({this.model = ''});

  factory OfflineFireRedAsrCtcModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineFireRedAsrCtcModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineFireRedAsrCtcModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {'model': model};

  final String model;
}

/// Model files and prompt settings for FunASR-Nano.
class OfflineFunAsrNanoModelConfig {
  const OfflineFunAsrNanoModelConfig({
    this.encoderAdaptor = '',
    this.llm = '',
    this.embedding = '',
    this.tokenizer = '',
    this.systemPrompt = 'You are a helpful assistant.',
    this.userPrompt = '语音转写：',
    this.maxNewTokens = 512,
    this.temperature = 1e-6,
    this.topP = 0.8,
    this.seed = 42,
    this.language = '',
    this.itn = 1,
    this.hotwords = '',
  });

  factory OfflineFunAsrNanoModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineFunAsrNanoModelConfig(
      encoderAdaptor: json['encoderAdaptor'] as String? ?? '',
      llm: json['llm'] as String? ?? '',
      embedding: json['embedding'] as String? ?? '',
      tokenizer: json['tokenizer'] as String? ?? '',
      systemPrompt: json['systemPrompt'] as String? ?? 'You are a helpful assistant.',
      userPrompt: json['userPrompt'] as String? ?? '语音转写：',
      maxNewTokens: json['maxNewTokens'] as int? ?? 512,
      temperature: (json['temperature'] as num?)?.toDouble() ?? 1e-6,
      topP: (json['topP'] as num?)?.toDouble() ?? 0.8,
      seed: json['seed'] as int? ?? 42,
      language: json['language'] as String? ?? '',
      itn: json['itn'] as int? ?? 1,
      hotwords: json['hotwords'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineFunAsrNanoModelConfig(encoderAdaptor: $encoderAdaptor, llm: $llm, embedding: $embedding, tokenizer: $tokenizer, systemPrompt: $systemPrompt, userPrompt: $userPrompt, maxNewTokens: $maxNewTokens, temperature: $temperature, topP: $topP, seed: $seed, language: $language, itn: $itn, hotwords: $hotwords)';
  }

  Map<String, dynamic> toJson() => {
    'encoderAdaptor': encoderAdaptor,
    'llm': llm,
    'embedding': embedding,
    'tokenizer': tokenizer,
    'systemPrompt': systemPrompt,
    'userPrompt': userPrompt,
    'maxNewTokens': maxNewTokens,
    'temperature': temperature,
    'topP': topP,
    'seed': seed,
    'language': language,
    'itn': itn,
    'hotwords': hotwords,
  };

  final String encoderAdaptor;
  final String llm;
  final String embedding;
  final String tokenizer;
  final String systemPrompt;
  final String userPrompt;
  final int maxNewTokens;
  final double temperature;
  final double topP;
  final int seed;
  final String language;
  final int itn;
  final String hotwords;
}

class OfflineQwen3AsrModelConfig {
  const OfflineQwen3AsrModelConfig({
    this.convFrontend = '',
    this.encoder = '',
    this.decoder = '',
    this.tokenizer = '',
    this.maxTotalLen = 512,
    this.maxNewTokens = 128,
    this.temperature = 1e-6,
    this.topP = 0.8,
    this.seed = 42,
    this.hotwords = '',
  });

  factory OfflineQwen3AsrModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineQwen3AsrModelConfig(
      convFrontend: json['convFrontend'] as String? ?? '',
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      tokenizer: json['tokenizer'] as String? ?? '',
      maxTotalLen: json['maxTotalLen'] as int? ?? 512,
      maxNewTokens: json['maxNewTokens'] as int? ?? 128,
      temperature: (json['temperature'] as num?)?.toDouble() ?? 1e-6,
      topP: (json['topP'] as num?)?.toDouble() ?? 0.8,
      seed: json['seed'] as int? ?? 42,
      hotwords: json['hotwords'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineQwen3AsrModelConfig(convFrontend: $convFrontend, encoder: $encoder, decoder: $decoder, tokenizer: $tokenizer, maxTotalLen: $maxTotalLen, maxNewTokens: $maxNewTokens, temperature: $temperature, topP: $topP, seed: $seed, hotwords: $hotwords)';
  }

  Map<String, dynamic> toJson() => {
    'convFrontend': convFrontend,
    'encoder': encoder,
    'decoder': decoder,
    'tokenizer': tokenizer,
    'maxTotalLen': maxTotalLen,
    'maxNewTokens': maxNewTokens,
    'temperature': temperature,
    'topP': topP,
    'seed': seed,
    'hotwords': hotwords,
  };

  final String convFrontend;
  final String encoder;
  final String decoder;
  final String tokenizer;
  final int maxTotalLen;
  final int maxNewTokens;
  final double temperature;
  final double topP;
  final int seed;
  final String hotwords;
}

/// Model files and options for an offline Whisper recognizer.
class OfflineWhisperModelConfig {
  const OfflineWhisperModelConfig({
    this.encoder = '',
    this.decoder = '',
    this.language = '',
    this.task = '',
    this.tailPaddings = -1,
    this.enableTokenTimestamps = false,
    this.enableSegmentTimestamps = false,
  });

  factory OfflineWhisperModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineWhisperModelConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      language: json['language'] as String? ?? '',
      task: json['task'] as String? ?? '',
      tailPaddings: json['tailPaddings'] as int? ?? -1,
      enableTokenTimestamps: json['enableTokenTimestamps'] as bool? ?? false,
      enableSegmentTimestamps:
          json['enableSegmentTimestamps'] as bool? ?? false,
    );
  }

  @override
  String toString() {
    return 'OfflineWhisperModelConfig(encoder: $encoder, decoder: $decoder, language: $language, task: $task, tailPaddings: $tailPaddings, enableTokenTimestamps: $enableTokenTimestamps, enableSegmentTimestamps: $enableSegmentTimestamps)';
  }

  Map<String, dynamic> toJson() => {
    'encoder': encoder,
    'decoder': decoder,
    'language': language,
    'task': task,
    'tailPaddings': tailPaddings,
    'enableTokenTimestamps': enableTokenTimestamps,
    'enableSegmentTimestamps': enableSegmentTimestamps,
  };

  final String encoder;
  final String decoder;
  final String language;
  final String task;
  final int tailPaddings;
  final bool enableTokenTimestamps;
  final bool enableSegmentTimestamps;
}

/// Model files and translation options for NeMo Canary.
class OfflineCanaryModelConfig {
  const OfflineCanaryModelConfig({
    this.encoder = '',
    this.decoder = '',
    this.srcLang = 'en',
    this.tgtLang = 'en',
    this.usePnc = true,
  });

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

/// Model files and text options for Cohere Transcribe.
class OfflineCohereTranscribeModelConfig {
  const OfflineCohereTranscribeModelConfig({
    this.encoder = '',
    this.decoder = '',
    this.language = '',
    this.usePunct = true,
    this.useItn = true,
  });

  factory OfflineCohereTranscribeModelConfig.fromJson(
    Map<String, dynamic> json,
  ) {
    return OfflineCohereTranscribeModelConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      language: json['language'] as String? ?? '',
      usePunct: json['usePunct'] as bool? ?? true,
      useItn: json['useItn'] as bool? ?? true,
    );
  }

  @override
  String toString() {
    return 'OfflineCohereTranscribeModelConfig(encoder: $encoder, decoder: $decoder, language: $language, usePunct: $usePunct, useItn: $useItn)';
  }

  Map<String, dynamic> toJson() => {
    'encoder': encoder,
    'decoder': decoder,
    'language': language,
    'usePunct': usePunct,
    'useItn': useItn,
  };

  final String encoder;
  final String decoder;
  final String language;
  final bool usePunct;
  final bool useItn;
}

/// Model files for the Fire-Red-ASR transducer recognizer.
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

  Map<String, dynamic> toJson() => {'encoder': encoder, 'decoder': decoder};

  final String encoder;
  final String decoder;
}

// For Moonshine v1, you need 4 models:
//  - preprocessor, encoder, uncachedDecoder, cachedDecoder
//
// For Moonshine v2, you need 2 models:
//  - encoder, mergedDecoder
/// Model files for Moonshine v1 or v2.
class OfflineMoonshineModelConfig {
  const OfflineMoonshineModelConfig({
    this.preprocessor = '',
    this.encoder = '',
    this.uncachedDecoder = '',
    this.cachedDecoder = '',
    this.mergedDecoder = '',
  });

  factory OfflineMoonshineModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineMoonshineModelConfig(
      preprocessor: json['preprocessor'] as String? ?? '',
      encoder: json['encoder'] as String? ?? '',
      uncachedDecoder: json['uncachedDecoder'] as String? ?? '',
      cachedDecoder: json['cachedDecoder'] as String? ?? '',
      mergedDecoder: json['mergedDecoder'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineMoonshineModelConfig(preprocessor: $preprocessor, encoder: $encoder, uncachedDecoder: $uncachedDecoder, cachedDecoder: $cachedDecoder, mergedDecoder: $mergedDecoder)';
  }

  Map<String, dynamic> toJson() => {
    'preprocessor': preprocessor,
    'encoder': encoder,
    'uncachedDecoder': uncachedDecoder,
    'cachedDecoder': cachedDecoder,
    'mergedDecoder': mergedDecoder,
  };

  final String preprocessor;
  final String encoder;
  final String uncachedDecoder;
  final String cachedDecoder;
  final String mergedDecoder;
}

/// Model files for an offline TDNN recognizer.
class OfflineTdnnModelConfig {
  const OfflineTdnnModelConfig({this.model = ''});

  factory OfflineTdnnModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTdnnModelConfig(model: json['model'] as String? ?? '');
  }

  @override
  String toString() {
    return 'OfflineTdnnModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {'model': model};

  final String model;
}

/// Model files and options for SenseVoice.
///
/// In the examples, this is typically paired with the
/// `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8` package.
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

/// Optional external language model settings for offline ASR.
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

  Map<String, dynamic> toJson() => {'model': model, 'scale': scale};

  final String model;
  final double scale;
}

/// Aggregate model configuration for offline recognition.
///
/// In typical use, configure exactly one model family and set the shared
/// options such as [tokens], [provider], and [numThreads].
///
/// For NeMo Parakeet-style transducer models, set [modelType] to
/// `nemo_transducer`, matching the repository examples.
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
    this.omnilingual = const OfflineOmnilingualAsrCtcModelConfig(),
    this.medasr = const OfflineMedAsrCtcModelConfig(),
    this.funasrNano = const OfflineFunAsrNanoModelConfig(),
    this.fireRedAsrCtc = const OfflineFireRedAsrCtcModelConfig(),
    this.qwen3Asr = const OfflineQwen3AsrModelConfig(),
    this.cohereTranscribe = const OfflineCohereTranscribeModelConfig(),
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
              json['transducer'] as Map<String, dynamic>,
            )
          : const OfflineTransducerModelConfig(),
      paraformer: json['paraformer'] != null
          ? OfflineParaformerModelConfig.fromJson(
              json['paraformer'] as Map<String, dynamic>,
            )
          : const OfflineParaformerModelConfig(),
      nemoCtc: json['nemoCtc'] != null
          ? OfflineNemoEncDecCtcModelConfig.fromJson(
              json['nemoCtc'] as Map<String, dynamic>,
            )
          : const OfflineNemoEncDecCtcModelConfig(),
      whisper: json['whisper'] != null
          ? OfflineWhisperModelConfig.fromJson(
              json['whisper'] as Map<String, dynamic>,
            )
          : const OfflineWhisperModelConfig(),
      tdnn: json['tdnn'] != null
          ? OfflineTdnnModelConfig.fromJson(
              json['tdnn'] as Map<String, dynamic>,
            )
          : const OfflineTdnnModelConfig(),
      senseVoice: json['senseVoice'] != null
          ? OfflineSenseVoiceModelConfig.fromJson(
              json['senseVoice'] as Map<String, dynamic>,
            )
          : const OfflineSenseVoiceModelConfig(),
      moonshine: json['moonshine'] != null
          ? OfflineMoonshineModelConfig.fromJson(
              json['moonshine'] as Map<String, dynamic>,
            )
          : const OfflineMoonshineModelConfig(),
      fireRedAsr: json['fireRedAsr'] != null
          ? OfflineFireRedAsrModelConfig.fromJson(
              json['fireRedAsr'] as Map<String, dynamic>,
            )
          : const OfflineFireRedAsrModelConfig(),
      dolphin: json['dolphin'] != null
          ? OfflineDolphinModelConfig.fromJson(
              json['dolphin'] as Map<String, dynamic>,
            )
          : const OfflineDolphinModelConfig(),
      zipformerCtc: json['zipformerCtc'] != null
          ? OfflineZipformerCtcModelConfig.fromJson(
              json['zipformerCtc'] as Map<String, dynamic>,
            )
          : const OfflineZipformerCtcModelConfig(),
      canary: json['canary'] != null
          ? OfflineCanaryModelConfig.fromJson(
              json['canary'] as Map<String, dynamic>,
            )
          : const OfflineCanaryModelConfig(),
      wenetCtc: json['wenetCtc'] != null
          ? OfflineWenetCtcModelConfig.fromJson(
              json['wenetCtc'] as Map<String, dynamic>,
            )
          : const OfflineWenetCtcModelConfig(),
      omnilingual: json['omnilingual'] != null
          ? OfflineOmnilingualAsrCtcModelConfig.fromJson(
              json['omnilingual'] as Map<String, dynamic>,
            )
          : const OfflineOmnilingualAsrCtcModelConfig(),
      medasr: json['medasr'] != null
          ? OfflineMedAsrCtcModelConfig.fromJson(
              json['medasr'] as Map<String, dynamic>,
            )
          : const OfflineMedAsrCtcModelConfig(),
      funasrNano: json['funasrNano'] != null
          ? OfflineFunAsrNanoModelConfig.fromJson(
              json['funasrNano'] as Map<String, dynamic>,
            )
          : const OfflineFunAsrNanoModelConfig(),
      fireRedAsrCtc: json['fireRedAsrCtc'] != null
          ? OfflineFireRedAsrCtcModelConfig.fromJson(
              json['fireRedAsrCtc'] as Map<String, dynamic>,
            )
          : const OfflineFireRedAsrCtcModelConfig(),
      qwen3Asr: json['qwen3Asr'] != null
          ? OfflineQwen3AsrModelConfig.fromJson(
              json['qwen3Asr'] as Map<String, dynamic>,
            )
          : const OfflineQwen3AsrModelConfig(),
      cohereTranscribe: json['cohereTranscribe'] != null
          ? OfflineCohereTranscribeModelConfig.fromJson(
              json['cohereTranscribe'] as Map<String, dynamic>,
            )
          : const OfflineCohereTranscribeModelConfig(),
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
    return 'OfflineModelConfig(transducer: $transducer, paraformer: $paraformer, nemoCtc: $nemoCtc, whisper: $whisper, tdnn: $tdnn, senseVoice: $senseVoice, moonshine: $moonshine, fireRedAsr: $fireRedAsr, dolphin: $dolphin, zipformerCtc: $zipformerCtc, canary: $canary, wenetCtc: $wenetCtc, omnilingual: $omnilingual, medasr: $medasr, funasrNano: $funasrNano, fireRedAsrCtc: $fireRedAsrCtc, qwen3Asr: $qwen3Asr, cohereTranscribe: $cohereTranscribe, tokens: $tokens, numThreads: $numThreads, debug: $debug, provider: $provider, modelType: $modelType, modelingUnit: $modelingUnit, bpeVocab: $bpeVocab, telespeechCtc: $telespeechCtc)';
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
    'omnilingual': omnilingual.toJson(),
    'medasr': medasr.toJson(),
    'funasrNano': funasrNano.toJson(),
    'fireRedAsrCtc': fireRedAsrCtc.toJson(),
    'qwen3Asr': qwen3Asr.toJson(),
    'cohereTranscribe': cohereTranscribe.toJson(),
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
  final OfflineOmnilingualAsrCtcModelConfig omnilingual;
  final OfflineMedAsrCtcModelConfig medasr;
  final OfflineFunAsrNanoModelConfig funasrNano;
  final OfflineFireRedAsrCtcModelConfig fireRedAsrCtc;
  final OfflineQwen3AsrModelConfig qwen3Asr;
  final OfflineCohereTranscribeModelConfig cohereTranscribe;

  final String tokens;
  final int numThreads;
  final bool debug;
  final String provider;
  final String modelType;
  final String modelingUnit;
  final String bpeVocab;
  final String telespeechCtc;
}

/// Top-level configuration for [OfflineRecognizer].
///
/// This combines feature extraction, the selected model family, optional
/// language model settings, hotwords, grammar resources, and optional
/// homophone replacement resources.
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
      hr: json['hr'] != null
          ? HomophoneReplacerConfig.fromJson(json['hr'] as Map<String, dynamic>)
          : const HomophoneReplacerConfig(),
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

/// Recognition result returned by [OfflineRecognizer.getResult].
///
/// Some model families populate [lang], [emotion], or [event] in addition to
/// the decoded text and token timestamps.
class OfflineRecognizerResult {
  OfflineRecognizerResult({
    required this.text,
    required this.tokens,
    required this.timestamps,
    required this.lang,
    required this.emotion,
    required this.event,
  });

  factory OfflineRecognizerResult.fromJson(Map<String, dynamic> json) {
    return OfflineRecognizerResult(
      text: json['text'] as String? ?? '',
      tokens: (json['tokens'] as List?)?.map((e) => e as String).toList() ?? [],
      timestamps:
          (json['timestamps'] as List?)
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
