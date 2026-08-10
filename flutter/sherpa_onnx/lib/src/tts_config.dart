// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for offline TTS -- no FFI, works on all platforms.
import 'dart:typed_data';

/// Per-request generation options for [OfflineTts.generateWithConfig].
///
/// Use this when you need advanced generation controls such as zero-shot voice
/// cloning reference audio, explicit reference sample rate, or model-specific
/// values in [extra].
class OfflineTtsGenerationConfig {
  const OfflineTtsGenerationConfig({
    this.silenceScale = 0.2,
    this.speed = 1.0,
    this.sid = 0,
    this.referenceAudio,
    this.referenceSampleRate = 0,
    this.referenceText = '',
    this.numSteps = 5,
    this.extra = const {},
  });

  final double silenceScale;
  final double speed;
  final int sid;

  /// mono audio in [-1, 1]
  final Float32List? referenceAudio;
  final int referenceSampleRate;
  final String referenceText;
  final int numSteps;

  /// Extra model-specific attributes
  /// key: string
  /// value: string | int | double
  final Map<String, Object> extra;
}

/// VITS model configuration.
class OfflineTtsVitsModelConfig {
  const OfflineTtsVitsModelConfig({
    this.model = '',
    this.lexicon = '',
    this.tokens = '',
    this.dataDir = '',
    this.noiseScale = 0.667,
    this.noiseScaleW = 0.8,
    this.lengthScale = 1.0,
    this.dictDir = '',
  });

  factory OfflineTtsVitsModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsVitsModelConfig(
      model: json['model'] as String? ?? '',
      lexicon: json['lexicon'] as String? ?? '',
      tokens: json['tokens'] as String? ?? '',
      dataDir: json['dataDir'] as String? ?? '',
      noiseScale: (json['noiseScale'] as num?)?.toDouble() ?? 0.667,
      noiseScaleW: (json['noiseScaleW'] as num?)?.toDouble() ?? 0.8,
      lengthScale: (json['lengthScale'] as num?)?.toDouble() ?? 1.0,
    );
  }

  @override
  String toString() {
    return 'OfflineTtsVitsModelConfig(model: $model, lexicon: $lexicon, tokens: $tokens, dataDir: $dataDir, noiseScale: $noiseScale, noiseScaleW: $noiseScaleW, lengthScale: $lengthScale)';
  }

  Map<String, dynamic> toJson() => {
    'model': model,
    'lexicon': lexicon,
    'tokens': tokens,
    'dataDir': dataDir,
    'noiseScale': noiseScale,
    'noiseScaleW': noiseScaleW,
    'lengthScale': lengthScale,
  };

  final String model;
  final String lexicon;
  final String tokens;
  final String dataDir;
  final double noiseScale;
  final double noiseScaleW;
  final double lengthScale;
  final String dictDir; // unused
}

/// Matcha model configuration.
class OfflineTtsMatchaModelConfig {
  const OfflineTtsMatchaModelConfig({
    this.acousticModel = '',
    this.vocoder = '',
    this.lexicon = '',
    this.tokens = '',
    this.dataDir = '',
    this.noiseScale = 0.667,
    this.lengthScale = 1.0,
    this.dictDir = '',
  });

  factory OfflineTtsMatchaModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsMatchaModelConfig(
      acousticModel: json['acousticModel'] as String? ?? '',
      vocoder: json['vocoder'] as String? ?? '',
      lexicon: json['lexicon'] as String? ?? '',
      tokens: json['tokens'] as String? ?? '',
      dataDir: json['dataDir'] as String? ?? '',
      noiseScale: (json['noiseScale'] as num?)?.toDouble() ?? 0.667,
      lengthScale: (json['lengthScale'] as num?)?.toDouble() ?? 1.0,
    );
  }

  @override
  String toString() {
    return 'OfflineTtsMatchaModelConfig(acousticModel: $acousticModel, vocoder: $vocoder, lexicon: $lexicon, tokens: $tokens, dataDir: $dataDir, noiseScale: $noiseScale, lengthScale: $lengthScale)';
  }

  Map<String, dynamic> toJson() => {
    'acousticModel': acousticModel,
    'vocoder': vocoder,
    'lexicon': lexicon,
    'tokens': tokens,
    'dataDir': dataDir,
    'noiseScale': noiseScale,
    'lengthScale': lengthScale,
  };

  final String acousticModel;
  final String vocoder;
  final String lexicon;
  final String tokens;
  final String dataDir;
  final double noiseScale;
  final double lengthScale;
  final String dictDir; // unused
}

/// Kokoro model configuration.
class OfflineTtsKokoroModelConfig {
  const OfflineTtsKokoroModelConfig({
    this.model = '',
    this.voices = '',
    this.tokens = '',
    this.dataDir = '',
    this.lengthScale = 1.0,
    this.dictDir = '',
    this.lexicon = '',
    this.lang = '',
  });

  factory OfflineTtsKokoroModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsKokoroModelConfig(
      model: json['model'] as String? ?? '',
      voices: json['voices'] as String? ?? '',
      tokens: json['tokens'] as String? ?? '',
      dataDir: json['dataDir'] as String? ?? '',
      lengthScale: (json['lengthScale'] as num?)?.toDouble() ?? 1.0,
      lexicon: json['lexicon'] as String? ?? '',
      lang: json['lang'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineTtsKokoroModelConfig(model: $model, voices: $voices, tokens: $tokens, dataDir: $dataDir, lengthScale: $lengthScale, lexicon: $lexicon, lang: $lang)';
  }

  Map<String, dynamic> toJson() => {
    'model': model,
    'voices': voices,
    'tokens': tokens,
    'dataDir': dataDir,
    'lengthScale': lengthScale,
    'lexicon': lexicon,
    'lang': lang,
  };

  final String model;
  final String voices;
  final String tokens;
  final String dataDir;
  final double lengthScale;
  final String dictDir; // unused
  final String lexicon;
  final String lang;
}

/// Kitten model configuration.
class OfflineTtsKittenModelConfig {
  const OfflineTtsKittenModelConfig({
    this.model = '',
    this.voices = '',
    this.tokens = '',
    this.dataDir = '',
    this.lengthScale = 1.0,
  });

  factory OfflineTtsKittenModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsKittenModelConfig(
      model: json['model'] as String? ?? '',
      voices: json['voices'] as String? ?? '',
      tokens: json['tokens'] as String? ?? '',
      dataDir: json['dataDir'] as String? ?? '',
      lengthScale: (json['lengthScale'] as num?)?.toDouble() ?? 1.0,
    );
  }

  @override
  String toString() {
    return 'OfflineTtsKittenModelConfig(model: $model, voices: $voices, tokens: $tokens, dataDir: $dataDir, lengthScale: $lengthScale)';
  }

  Map<String, dynamic> toJson() => {
    'model': model,
    'voices': voices,
    'tokens': tokens,
    'dataDir': dataDir,
    'lengthScale': lengthScale,
  };

  final String model;
  final String voices;
  final String tokens;
  final String dataDir;
  final double lengthScale;
}

/// ZipVoice model configuration.
class OfflineTtsZipVoiceModelConfig {
  const OfflineTtsZipVoiceModelConfig({
    this.tokens = '',
    this.encoder = '',
    this.decoder = '',
    this.vocoder = '',
    this.dataDir = '',
    this.lexicon = '',
    this.featScale = 0.1,
    this.tShift = 0.5,
    this.targetRms = 0.1,
    this.guidanceScale = 1.0,
  });

  factory OfflineTtsZipVoiceModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsZipVoiceModelConfig(
      tokens: json['tokens'] as String? ?? '',
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      vocoder: json['vocoder'] as String? ?? '',
      dataDir: json['dataDir'] as String? ?? '',
      lexicon: json['lexicon'] as String? ?? '',
      featScale: (json['featScale'] as num?)?.toDouble() ?? 0.1,
      tShift: (json['tShift'] as num?)?.toDouble() ?? 0.5,
      targetRms: (json['targetRms'] as num?)?.toDouble() ?? 0.1,
      guidanceScale: (json['guidanceScale'] as num?)?.toDouble() ?? 1.0,
    );
  }

  @override
  String toString() {
    return 'OfflineTtsZipVoiceModelConfig(tokens: $tokens, encoder: $encoder, decoder: $decoder, vocoder: $vocoder, dataDir: $dataDir, lexicon: $lexicon, featScale: $featScale, tShift: $tShift, targetRms: $targetRms, guidanceScale: $guidanceScale)';
  }

  Map<String, dynamic> toJson() => {
    'tokens': tokens,
    'encoder': encoder,
    'decoder': decoder,
    'vocoder': vocoder,
    'dataDir': dataDir,
    'lexicon': lexicon,
    'featScale': featScale,
    'tShift': tShift,
    'targetRms': targetRms,
    'guidanceScale': guidanceScale,
  };

  final String tokens;
  final String encoder;
  final String decoder;
  final String vocoder;
  final String dataDir;
  final String lexicon;
  final double featScale;
  final double tShift;
  final double targetRms;
  final double guidanceScale;
}

/// Pocket TTS model configuration.
///
/// This family supports zero-shot voice cloning with a reference waveform.
class OfflineTtsPocketModelConfig {
  const OfflineTtsPocketModelConfig({
    this.lmFlow = '',
    this.lmMain = '',
    this.encoder = '',
    this.decoder = '',
    this.textConditioner = '',
    this.vocabJson = '',
    this.tokenScoresJson = '',
    this.voiceEmbeddingCacheCapacity = 50,
  });

  factory OfflineTtsPocketModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsPocketModelConfig(
      lmFlow: json['lmFlow'] as String? ?? '',
      lmMain: json['lmMain'] as String? ?? '',
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      textConditioner: json['textConditioner'] as String? ?? '',
      vocabJson: json['vocabJson'] as String? ?? '',
      tokenScoresJson: json['tokenScoresJson'] as String? ?? '',
      voiceEmbeddingCacheCapacity:
          json['voiceEmbeddingCacheCapacity'] as int? ?? 50,
    );
  }

  Map<String, dynamic> toJson() => {
    'lmFlow': lmFlow,
    'lmMain': lmMain,
    'encoder': encoder,
    'decoder': decoder,
    'textConditioner': textConditioner,
    'vocabJson': vocabJson,
    'tokenScoresJson': tokenScoresJson,
    'voiceEmbeddingCacheCapacity': voiceEmbeddingCacheCapacity,
  };

  @override
  String toString() {
    return 'OfflineTtsPocketModelConfig(lmFlow: $lmFlow, lmMain: $lmMain, encoder: $encoder, decoder: $decoder, textConditioner: $textConditioner, vocabJson: $vocabJson, tokenScoresJson: $tokenScoresJson, voiceEmbeddingCacheCapacity: $voiceEmbeddingCacheCapacity)';
  }

  final String lmFlow;
  final String lmMain;
  final String encoder;
  final String decoder;
  final String textConditioner;
  final String vocabJson;
  final String tokenScoresJson;
  final int voiceEmbeddingCacheCapacity;
}

/// Supertonic model configuration.
class OfflineTtsSupertonicModelConfig {
  const OfflineTtsSupertonicModelConfig({
    this.durationPredictor = '',
    this.textEncoder = '',
    this.vectorEstimator = '',
    this.vocoder = '',
    this.ttsJson = '',
    this.unicodeIndexer = '',
    this.voiceStyle = '',
  });

  factory OfflineTtsSupertonicModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsSupertonicModelConfig(
      durationPredictor: json['durationPredictor'] as String? ?? '',
      textEncoder: json['textEncoder'] as String? ?? '',
      vectorEstimator: json['vectorEstimator'] as String? ?? '',
      vocoder: json['vocoder'] as String? ?? '',
      ttsJson: json['ttsJson'] as String? ?? '',
      unicodeIndexer: json['unicodeIndexer'] as String? ?? '',
      voiceStyle: json['voiceStyle'] as String? ?? '',
    );
  }

  Map<String, dynamic> toJson() => {
    'durationPredictor': durationPredictor,
    'textEncoder': textEncoder,
    'vectorEstimator': vectorEstimator,
    'vocoder': vocoder,
    'ttsJson': ttsJson,
    'unicodeIndexer': unicodeIndexer,
    'voiceStyle': voiceStyle,
  };

  @override
  String toString() {
    return 'OfflineTtsSupertonicModelConfig(durationPredictor: $durationPredictor, textEncoder: $textEncoder, vectorEstimator: $vectorEstimator, vocoder: $vocoder, ttsJson: $ttsJson, unicodeIndexer: $unicodeIndexer, voiceStyle: $voiceStyle)';
  }

  final String durationPredictor;
  final String textEncoder;
  final String vectorEstimator;
  final String vocoder;
  final String ttsJson;
  final String unicodeIndexer;
  final String voiceStyle;
}

/// Aggregate model configuration for offline TTS.
///
/// Configure exactly one model family for a typical setup and set the shared
/// runtime options such as [numThreads] and [provider].
class OfflineTtsModelConfig {
  const OfflineTtsModelConfig({
    this.vits = const OfflineTtsVitsModelConfig(),
    this.matcha = const OfflineTtsMatchaModelConfig(),
    this.kokoro = const OfflineTtsKokoroModelConfig(),
    this.kitten = const OfflineTtsKittenModelConfig(),
    this.zipvoice = const OfflineTtsZipVoiceModelConfig(),
    this.pocket = const OfflineTtsPocketModelConfig(),
    this.supertonic = const OfflineTtsSupertonicModelConfig(),
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
  });

  factory OfflineTtsModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsModelConfig(
      vits: OfflineTtsVitsModelConfig.fromJson(
        json['vits'] as Map<String, dynamic>? ?? const {},
      ),
      matcha: OfflineTtsMatchaModelConfig.fromJson(
        json['matcha'] as Map<String, dynamic>? ?? const {},
      ),
      kokoro: OfflineTtsKokoroModelConfig.fromJson(
        json['kokoro'] as Map<String, dynamic>? ?? const {},
      ),
      kitten: OfflineTtsKittenModelConfig.fromJson(
        json['kitten'] as Map<String, dynamic>? ?? const {},
      ),
      zipvoice: OfflineTtsZipVoiceModelConfig.fromJson(
        json['zipvoice'] as Map<String, dynamic>? ?? const {},
      ),
      pocket: OfflineTtsPocketModelConfig.fromJson(
        json['pocket'] as Map<String, dynamic>? ?? const {},
      ),
      supertonic: OfflineTtsSupertonicModelConfig.fromJson(
        json['supertonic'] as Map<String, dynamic>? ?? const {},
      ),
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'OfflineTtsModelConfig(vits: $vits, matcha: $matcha, kokoro: $kokoro, kitten: $kitten, zipvoice: $zipvoice, pocket: $pocket, supertonic: $supertonic, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
    'vits': vits.toJson(),
    'matcha': matcha.toJson(),
    'kokoro': kokoro.toJson(),
    'kitten': kitten.toJson(),
    'zipvoice': zipvoice.toJson(),
    'pocket': pocket.toJson(),
    'supertonic': supertonic.toJson(),
    'numThreads': numThreads,
    'debug': debug,
    'provider': provider,
  };

  final OfflineTtsVitsModelConfig vits;
  final OfflineTtsMatchaModelConfig matcha;
  final OfflineTtsKokoroModelConfig kokoro;
  final OfflineTtsKittenModelConfig kitten;
  final OfflineTtsZipVoiceModelConfig zipvoice;
  final OfflineTtsPocketModelConfig pocket;
  final OfflineTtsSupertonicModelConfig supertonic;
  final int numThreads;
  final bool debug;
  final String provider;
}

/// Top-level configuration for [OfflineTts].
class OfflineTtsConfig {
  const OfflineTtsConfig({
    required this.model,
    this.ruleFsts = '',
    this.maxNumSenetences = 1,
    this.ruleFars = '',
    this.silenceScale = 0.2,
  });

  factory OfflineTtsConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsConfig(
      model: OfflineTtsModelConfig.fromJson(
        json['model'] as Map<String, dynamic>,
      ),
      ruleFsts: json['ruleFsts'] as String? ?? '',
      maxNumSenetences: json['maxNumSenetences'] as int? ?? 1,
      ruleFars: json['ruleFars'] as String? ?? '',
      silenceScale: (json['silenceScale'] as num?)?.toDouble() ?? 0.2,
    );
  }

  @override
  String toString() {
    return 'OfflineTtsConfig(model: $model, ruleFsts: $ruleFsts, maxNumSenetences: $maxNumSenetences, ruleFars: $ruleFars, silenceScale: $silenceScale)';
  }

  Map<String, dynamic> toJson() => {
    'model': model.toJson(),
    'ruleFsts': ruleFsts,
    'maxNumSenetences': maxNumSenetences,
    'ruleFars': ruleFars,
    'silenceScale': silenceScale,
  };

  final OfflineTtsModelConfig model;
  final String ruleFsts;
  final int maxNumSenetences;
  final String ruleFars;
  final double silenceScale;
}

/// Audio generated by [OfflineTts].
class GeneratedAudio {
  GeneratedAudio({required this.samples, required this.sampleRate});

  final Float32List samples;
  final int sampleRate;
}
