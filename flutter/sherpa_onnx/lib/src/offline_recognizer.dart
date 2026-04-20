// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './feature_config.dart';
import './homophone_replacer_config.dart';
import './offline_stream.dart';
import './sherpa_onnx_bindings.dart';
import './utils.dart';

/// Offline speech recognition.
///
/// This module covers non-streaming ASR model families such as transducer,
/// Paraformer, Whisper, SenseVoice, Moonshine, Canary, Fire-Red-ASR, WeNet,
/// Omnilingual-ASR, TeleSpeech-CTC, FunASR-Nano, and several CTC variants.
///
/// See `dart-api-examples/non-streaming-asr/bin/` for concrete usage,
/// including `sense-voice.dart`, `whisper.dart`, `nemo-transducer.dart`,
/// `moonshine_v2.dart`, and `fire-red-asr-ctc.dart`.
///
/// Example:
///
/// ```dart
/// final whisper = OfflineWhisperModelConfig(
///   encoder: './sherpa-onnx-whisper-tiny/encoder.int8.onnx',
///   decoder: './sherpa-onnx-whisper-tiny/decoder.int8.onnx',
/// );
///
/// final model = OfflineModelConfig(
///   whisper: whisper,
///   tokens: './sherpa-onnx-whisper-tiny/tokens.txt',
///   modelType: 'whisper',
///   numThreads: 1,
/// );
///
/// final recognizer = OfflineRecognizer(OfflineRecognizerConfig(model: model));
/// final wave = readWave('./test.wav');
/// final stream = recognizer.createStream();
/// stream.acceptWaveform(samples: wave.samples, sampleRate: wave.sampleRate);
/// recognizer.decode(stream);
/// print(recognizer.getResult(stream).text);
/// stream.free();
/// recognizer.free();
/// ```

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
      systemPrompt: json['systemPrompt'] as String? ?? '',
      userPrompt: json['userPrompt'] as String? ?? '',
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

/// Offline speech recognizer.
///
/// Create one from an [OfflineRecognizerConfig], then create an
/// [OfflineStream], feed waveform samples, call [decode], and fetch the final
/// hypothesis with [getResult].
class OfflineRecognizer {
  OfflineRecognizer.fromPtr({required this.ptr, required this.config});

  OfflineRecognizer._({required this.ptr, required this.config});

  /// Release the native recognizer.
  void free() {
    if (SherpaOnnxBindings.destroyOfflineRecognizer == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroyOfflineRecognizer?.call(ptr);
    ptr = nullptr;
  }

  /// The user is responsible to call the OfflineRecognizer.free()
  /// method of the returned instance to avoid memory leak.

  /// Create a recognizer from [config].
  factory OfflineRecognizer(OfflineRecognizerConfig config) {
    final c = convertConfig(config);

    if (SherpaOnnxBindings.createOfflineRecognizer == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final ptr = SherpaOnnxBindings.createOfflineRecognizer?.call(c) ?? nullptr;

    if (ptr == nullptr) {
      throw Exception(
        "Failed to create offline recognizer. Please check your config",
      );
    }

    freeConfig(c);

    return OfflineRecognizer._(ptr: ptr, config: config);
  }

  /// Replace the runtime configuration.
  void setConfig(OfflineRecognizerConfig config) {
    if (SherpaOnnxBindings.offlineRecognizerSetConfig == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }

    final c = convertConfig(config);

    SherpaOnnxBindings.offlineRecognizerSetConfig?.call(ptr, c);

    freeConfig(c);
    // we don't update this.config
  }

  static Pointer<SherpaOnnxOfflineRecognizerConfig> convertConfig(
    OfflineRecognizerConfig config,
  ) {
    final c = calloc<SherpaOnnxOfflineRecognizerConfig>();

    c.ref.feat.sampleRate = config.feat.sampleRate;
    c.ref.feat.featureDim = config.feat.featureDim;

    // transducer
    c.ref.model.transducer.encoder = config.model.transducer.encoder
        .toNativeUtf8();
    c.ref.model.transducer.decoder = config.model.transducer.decoder
        .toNativeUtf8();
    c.ref.model.transducer.joiner = config.model.transducer.joiner
        .toNativeUtf8();

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
    c.ref.model.whisper.enableTokenTimestamps =
        config.model.whisper.enableTokenTimestamps ? 1 : 0;
    c.ref.model.whisper.enableSegmentTimestamps =
        config.model.whisper.enableSegmentTimestamps ? 1 : 0;

    c.ref.model.tdnn.model = config.model.tdnn.model.toNativeUtf8();

    c.ref.model.senseVoice.model = config.model.senseVoice.model.toNativeUtf8();

    c.ref.model.senseVoice.language = config.model.senseVoice.language
        .toNativeUtf8();

    c.ref.model.senseVoice.useInverseTextNormalization =
        config.model.senseVoice.useInverseTextNormalization ? 1 : 0;

    c.ref.model.moonshine.preprocessor = config.model.moonshine.preprocessor
        .toNativeUtf8();
    c.ref.model.moonshine.encoder = config.model.moonshine.encoder
        .toNativeUtf8();
    c.ref.model.moonshine.uncachedDecoder = config
        .model
        .moonshine
        .uncachedDecoder
        .toNativeUtf8();
    c.ref.model.moonshine.cachedDecoder = config.model.moonshine.cachedDecoder
        .toNativeUtf8();
    c.ref.model.moonshine.mergedDecoder = config.model.moonshine.mergedDecoder
        .toNativeUtf8();

    // FireRedAsr
    c.ref.model.fireRedAsr.encoder = config.model.fireRedAsr.encoder
        .toNativeUtf8();
    c.ref.model.fireRedAsr.decoder = config.model.fireRedAsr.decoder
        .toNativeUtf8();

    c.ref.model.dolphin.model = config.model.dolphin.model.toNativeUtf8();
    c.ref.model.zipformerCtc.model = config.model.zipformerCtc.model
        .toNativeUtf8();

    c.ref.model.canary.encoder = config.model.canary.encoder.toNativeUtf8();
    c.ref.model.canary.decoder = config.model.canary.decoder.toNativeUtf8();
    c.ref.model.canary.srcLang = config.model.canary.srcLang.toNativeUtf8();
    c.ref.model.canary.tgtLang = config.model.canary.tgtLang.toNativeUtf8();
    c.ref.model.canary.usePnc = config.model.canary.usePnc ? 1 : 0;

    c.ref.model.wenetCtc.model = config.model.wenetCtc.model.toNativeUtf8();
    c.ref.model.omnilingual.model = config.model.omnilingual.model
        .toNativeUtf8();
    c.ref.model.medasr.model = config.model.medasr.model.toNativeUtf8();

    c.ref.model.funasrNano.encoderAdaptor = config
        .model
        .funasrNano
        .encoderAdaptor
        .toNativeUtf8();
    c.ref.model.funasrNano.llm = config.model.funasrNano.llm.toNativeUtf8();
    c.ref.model.funasrNano.embedding = config.model.funasrNano.embedding
        .toNativeUtf8();
    c.ref.model.funasrNano.tokenizer = config.model.funasrNano.tokenizer
        .toNativeUtf8();
    c.ref.model.funasrNano.systemPrompt = config.model.funasrNano.systemPrompt
        .toNativeUtf8();
    c.ref.model.funasrNano.userPrompt = config.model.funasrNano.userPrompt
        .toNativeUtf8();
    c.ref.model.funasrNano.maxNewTokens = config.model.funasrNano.maxNewTokens;
    c.ref.model.funasrNano.temperature = config.model.funasrNano.temperature;
    c.ref.model.funasrNano.topP = config.model.funasrNano.topP;
    c.ref.model.funasrNano.seed = config.model.funasrNano.seed;
    c.ref.model.funasrNano.language = config.model.funasrNano.language
        .toNativeUtf8();
    c.ref.model.funasrNano.itn = config.model.funasrNano.itn;
    c.ref.model.funasrNano.hotwords = config.model.funasrNano.hotwords
        .toNativeUtf8();

    c.ref.model.fireRedAsrCtc.model = config.model.fireRedAsrCtc.model
        .toNativeUtf8();

    c.ref.model.qwen3Asr.convFrontend = config.model.qwen3Asr.convFrontend
        .toNativeUtf8();
    c.ref.model.qwen3Asr.encoder = config.model.qwen3Asr.encoder
        .toNativeUtf8();
    c.ref.model.qwen3Asr.decoder = config.model.qwen3Asr.decoder
        .toNativeUtf8();
    c.ref.model.qwen3Asr.tokenizer = config.model.qwen3Asr.tokenizer
        .toNativeUtf8();
    c.ref.model.qwen3Asr.maxTotalLen = config.model.qwen3Asr.maxTotalLen;
    c.ref.model.qwen3Asr.maxNewTokens = config.model.qwen3Asr.maxNewTokens;
    c.ref.model.qwen3Asr.temperature = config.model.qwen3Asr.temperature;
    c.ref.model.qwen3Asr.topP = config.model.qwen3Asr.topP;
    c.ref.model.qwen3Asr.seed = config.model.qwen3Asr.seed;
    c.ref.model.qwen3Asr.hotwords = config.model.qwen3Asr.hotwords
        .toNativeUtf8();

    c.ref.model.cohereTranscribe.encoder = config
        .model
        .cohereTranscribe
        .encoder
        .toNativeUtf8();
    c.ref.model.cohereTranscribe.decoder = config
        .model
        .cohereTranscribe
        .decoder
        .toNativeUtf8();
    c.ref.model.cohereTranscribe.language = config
        .model
        .cohereTranscribe
        .language
        .toNativeUtf8();
    c.ref.model.cohereTranscribe.usePunct =
        config.model.cohereTranscribe.usePunct ? 1 : 0;
    c.ref.model.cohereTranscribe.useItn =
        config.model.cohereTranscribe.useItn ? 1 : 0;

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

    c.ref.hr.lexicon = config.hr.lexicon.toNativeUtf8();
    c.ref.hr.ruleFsts = config.hr.ruleFsts.toNativeUtf8();

    return c;
  }

  static void freeConfig(Pointer<SherpaOnnxOfflineRecognizerConfig> c) {
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
    calloc.free(c.ref.model.qwen3Asr.hotwords);
    calloc.free(c.ref.model.qwen3Asr.tokenizer);
    calloc.free(c.ref.model.qwen3Asr.decoder);
    calloc.free(c.ref.model.qwen3Asr.encoder);
    calloc.free(c.ref.model.qwen3Asr.convFrontend);
    calloc.free(c.ref.model.cohereTranscribe.language);
    calloc.free(c.ref.model.cohereTranscribe.decoder);
    calloc.free(c.ref.model.cohereTranscribe.encoder);
    calloc.free(c.ref.model.fireRedAsrCtc.model);
    calloc.free(c.ref.model.funasrNano.hotwords);
    calloc.free(c.ref.model.funasrNano.language);
    calloc.free(c.ref.model.funasrNano.userPrompt);
    calloc.free(c.ref.model.funasrNano.systemPrompt);
    calloc.free(c.ref.model.funasrNano.tokenizer);
    calloc.free(c.ref.model.funasrNano.embedding);
    calloc.free(c.ref.model.funasrNano.llm);
    calloc.free(c.ref.model.funasrNano.encoderAdaptor);
    calloc.free(c.ref.model.medasr.model);
    calloc.free(c.ref.model.omnilingual.model);
    calloc.free(c.ref.model.wenetCtc.model);
    calloc.free(c.ref.model.canary.tgtLang);
    calloc.free(c.ref.model.canary.srcLang);
    calloc.free(c.ref.model.canary.decoder);
    calloc.free(c.ref.model.canary.encoder);
    calloc.free(c.ref.model.zipformerCtc.model);
    calloc.free(c.ref.model.dolphin.model);
    calloc.free(c.ref.model.fireRedAsr.decoder);
    calloc.free(c.ref.model.fireRedAsr.encoder);
    calloc.free(c.ref.model.moonshine.mergedDecoder);
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
  /// Create an offline stream.
  OfflineStream createStream() {
    if (SherpaOnnxBindings.createOfflineStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      throw Exception("Failed to create offline stream");
    }

    final p = SherpaOnnxBindings.createOfflineStream?.call(ptr) ?? nullptr;

    if (p == nullptr) {
      throw Exception("Failed to create offline stream");
    }

    return OfflineStream(ptr: p);
  }

  /// Decode one stream.
  void decode(OfflineStream stream) {
    if (SherpaOnnxBindings.decodeOfflineStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return;
    }

    SherpaOnnxBindings.decodeOfflineStream?.call(ptr, stream.ptr);
  }

  /// Fetch the current recognition result for [stream].
  OfflineRecognizerResult getResult(OfflineStream stream) {
    if (SherpaOnnxBindings.getOfflineStreamResultAsJson == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return OfflineRecognizerResult(
        text: '',
        tokens: [],
        timestamps: [],
        lang: '',
        emotion: '',
        event: '',
      );
    }

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
        event: '',
      );
    }

    final parsedJson = jsonDecode(toDartString(json));

    SherpaOnnxBindings.destroyOfflineStreamResultJson?.call(json);

    return OfflineRecognizerResult(
      text: parsedJson['text'],
      tokens: List<String>.from(parsedJson['tokens']),
      timestamps: List<double>.from(parsedJson['timestamps']),
      lang: parsedJson['lang'],
      emotion: parsedJson['emotion'],
      event: parsedJson['event'],
    );
  }

  Pointer<SherpaOnnxOfflineRecognizer> ptr;
  OfflineRecognizerConfig config;
}
