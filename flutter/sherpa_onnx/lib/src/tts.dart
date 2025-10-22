// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

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

class OfflineTtsZipVoiceModelConfig {
  const OfflineTtsZipVoiceModelConfig({
    this.tokens = '',
    this.textModel = '',
    this.flowMatchingModel = '',
    this.vocoder = '',
    this.dataDir = '',
    this.pinyinDict = '',
    this.featScale = 0.1,
    this.tShift = 0.5,
    this.targetRms = 0.1,
    this.guidanceScale = 1.0,
  });

  factory OfflineTtsZipVoiceModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsZipVoiceModelConfig(
      tokens: json['tokens'] as String? ?? '',
      textModel: json['textModel'] as String? ?? '',
      flowMatchingModel: json['flowMatchingModel'] as String? ?? '',
      vocoder: json['vocoder'] as String? ?? '',
      dataDir: json['dataDir'] as String? ?? '',
      pinyinDict: json['pinyinDict'] as String? ?? '',
      featScale: (json['featScale'] as num?)?.toDouble() ?? 0.1,
      tShift: (json['tShift'] as num?)?.toDouble() ?? 0.5,
      targetRms: (json['targetRms'] as num?)?.toDouble() ?? 0.1,
      guidanceScale: (json['guidanceScale'] as num?)?.toDouble() ?? 1.0,
    );
  }

  @override
  String toString() {
    return 'OfflineTtsZipVoiceModelConfig(tokens: $tokens, textModel: $textModel, flowMatchingModel: $flowMatchingModel, vocoder: $vocoder, dataDir: $dataDir, pinyinDict: $pinyinDict, featScale: $featScale, tShift: $tShift, targetRms: $targetRms, guidanceScale: $guidanceScale)';
  }

  Map<String, dynamic> toJson() => {
        'tokens': tokens,
        'textModel': textModel,
        'flowMatchingModel': flowMatchingModel,
        'vocoder': vocoder,
        'dataDir': dataDir,
        'pinyinDict': pinyinDict,
        'featScale': featScale,
        'tShift': tShift,
        'targetRms': targetRms,
        'guidanceScale': guidanceScale,
      };

  final String tokens;
  final String textModel;
  final String flowMatchingModel;
  final String vocoder;
  final String dataDir;
  final String pinyinDict;
  final double featScale;
  final double tShift;
  final double targetRms;
  final double guidanceScale;
}

class OfflineTtsModelConfig {
  const OfflineTtsModelConfig({
    this.vits = const OfflineTtsVitsModelConfig(),
    this.matcha = const OfflineTtsMatchaModelConfig(),
    this.kokoro = const OfflineTtsKokoroModelConfig(),
    this.kitten = const OfflineTtsKittenModelConfig(),
    this.zipvoice = const OfflineTtsZipVoiceModelConfig(),
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
  });

  factory OfflineTtsModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineTtsModelConfig(
      vits: OfflineTtsVitsModelConfig.fromJson(
          json['vits'] as Map<String, dynamic>? ?? const {}),
      matcha: OfflineTtsMatchaModelConfig.fromJson(
          json['matcha'] as Map<String, dynamic>? ?? const {}),
      kokoro: OfflineTtsKokoroModelConfig.fromJson(
          json['kokoro'] as Map<String, dynamic>? ?? const {}),
      kitten: OfflineTtsKittenModelConfig.fromJson(
          json['kitten'] as Map<String, dynamic>? ?? const {}),
      zipvoice: OfflineTtsZipVoiceModelConfig.fromJson(
          json['zipvoice'] as Map<String, dynamic>? ?? const {}),
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'OfflineTtsModelConfig(vits: $vits, matcha: $matcha, kokoro: $kokoro, kitten: $kitten, zipvoice: $zipvoice, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'vits': vits.toJson(),
        'matcha': matcha.toJson(),
        'kokoro': kokoro.toJson(),
        'kitten': kitten.toJson(),
        'zipvoice': zipvoice.toJson(),
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final OfflineTtsVitsModelConfig vits;
  final OfflineTtsMatchaModelConfig matcha;
  final OfflineTtsKokoroModelConfig kokoro;
  final OfflineTtsKittenModelConfig kitten;
  final OfflineTtsZipVoiceModelConfig zipvoice;
  final int numThreads;
  final bool debug;
  final String provider;
}

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
      model:
          OfflineTtsModelConfig.fromJson(json['model'] as Map<String, dynamic>),
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

class GeneratedAudio {
  GeneratedAudio({
    required this.samples,
    required this.sampleRate,
  });

  final Float32List samples;
  final int sampleRate;
}

class OfflineTts {
  OfflineTts.fromPtr({required this.ptr, required this.config});

  OfflineTts._({required this.ptr, required this.config});

  /// The user is responsible to call the OfflineTts.free()
  /// method of the returned instance to avoid memory leak.
  factory OfflineTts(OfflineTtsConfig config) {
    if (SherpaOnnxBindings.createOfflineTts == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final c = calloc<SherpaOnnxOfflineTtsConfig>();
    c.ref.model.vits.model = config.model.vits.model.toNativeUtf8();
    c.ref.model.vits.lexicon = config.model.vits.lexicon.toNativeUtf8();
    c.ref.model.vits.tokens = config.model.vits.tokens.toNativeUtf8();
    c.ref.model.vits.dataDir = config.model.vits.dataDir.toNativeUtf8();
    c.ref.model.vits.noiseScale = config.model.vits.noiseScale;
    c.ref.model.vits.noiseScaleW = config.model.vits.noiseScaleW;
    c.ref.model.vits.lengthScale = config.model.vits.lengthScale;

    c.ref.model.matcha.acousticModel =
        config.model.matcha.acousticModel.toNativeUtf8();
    c.ref.model.matcha.vocoder = config.model.matcha.vocoder.toNativeUtf8();
    c.ref.model.matcha.lexicon = config.model.matcha.lexicon.toNativeUtf8();
    c.ref.model.matcha.tokens = config.model.matcha.tokens.toNativeUtf8();
    c.ref.model.matcha.dataDir = config.model.matcha.dataDir.toNativeUtf8();
    c.ref.model.matcha.noiseScale = config.model.matcha.noiseScale;
    c.ref.model.matcha.lengthScale = config.model.matcha.lengthScale;

    c.ref.model.kokoro.model = config.model.kokoro.model.toNativeUtf8();
    c.ref.model.kokoro.voices = config.model.kokoro.voices.toNativeUtf8();
    c.ref.model.kokoro.tokens = config.model.kokoro.tokens.toNativeUtf8();
    c.ref.model.kokoro.dataDir = config.model.kokoro.dataDir.toNativeUtf8();
    c.ref.model.kokoro.lengthScale = config.model.kokoro.lengthScale;
    c.ref.model.kokoro.lexicon = config.model.kokoro.lexicon.toNativeUtf8();
    c.ref.model.kokoro.lang = config.model.kokoro.lang.toNativeUtf8();

    c.ref.model.kitten.model = config.model.kitten.model.toNativeUtf8();
    c.ref.model.kitten.voices = config.model.kitten.voices.toNativeUtf8();
    c.ref.model.kitten.tokens = config.model.kitten.tokens.toNativeUtf8();
    c.ref.model.kitten.dataDir = config.model.kitten.dataDir.toNativeUtf8();
    c.ref.model.kitten.lengthScale = config.model.kitten.lengthScale;

    c.ref.model.zipvoice.tokens = config.model.zipvoice.tokens.toNativeUtf8();
    c.ref.model.zipvoice.textModel = config.model.zipvoice.textModel.toNativeUtf8();
    c.ref.model.zipvoice.flowMatchingModel = config.model.zipvoice.flowMatchingModel.toNativeUtf8();
    c.ref.model.zipvoice.vocoder = config.model.zipvoice.vocoder.toNativeUtf8();
    c.ref.model.zipvoice.dataDir = config.model.zipvoice.dataDir.toNativeUtf8();
    c.ref.model.zipvoice.pinyinDict = config.model.zipvoice.pinyinDict.toNativeUtf8();
    c.ref.model.zipvoice.featScale = config.model.zipvoice.featScale;
    c.ref.model.zipvoice.tShift = config.model.zipvoice.tShift;
    c.ref.model.zipvoice.targetRms = config.model.zipvoice.targetRms;
    c.ref.model.zipvoice.guidanceScale = config.model.zipvoice.guidanceScale;

    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.debug = config.model.debug ? 1 : 0;
    c.ref.model.provider = config.model.provider.toNativeUtf8();

    c.ref.ruleFsts = config.ruleFsts.toNativeUtf8();
    c.ref.maxNumSenetences = config.maxNumSenetences;
    c.ref.ruleFars = config.ruleFars.toNativeUtf8();
    c.ref.silenceScale = config.silenceScale;

    final ptr = SherpaOnnxBindings.createOfflineTts?.call(c) ?? nullptr;

    calloc.free(c.ref.ruleFars);
    calloc.free(c.ref.ruleFsts);
    calloc.free(c.ref.model.provider);

    calloc.free(c.ref.model.zipvoice.pinyinDict);
    calloc.free(c.ref.model.zipvoice.dataDir);
    calloc.free(c.ref.model.zipvoice.vocoder);
    calloc.free(c.ref.model.zipvoice.flowMatchingModel);
    calloc.free(c.ref.model.zipvoice.textModel);
    calloc.free(c.ref.model.zipvoice.tokens);

    calloc.free(c.ref.model.kitten.dataDir);
    calloc.free(c.ref.model.kitten.tokens);
    calloc.free(c.ref.model.kitten.voices);
    calloc.free(c.ref.model.kitten.model);

    calloc.free(c.ref.model.kokoro.lang);
    calloc.free(c.ref.model.kokoro.lexicon);
    calloc.free(c.ref.model.kokoro.dataDir);
    calloc.free(c.ref.model.kokoro.tokens);
    calloc.free(c.ref.model.kokoro.voices);
    calloc.free(c.ref.model.kokoro.model);

    calloc.free(c.ref.model.matcha.dataDir);
    calloc.free(c.ref.model.matcha.tokens);
    calloc.free(c.ref.model.matcha.lexicon);
    calloc.free(c.ref.model.matcha.vocoder);
    calloc.free(c.ref.model.matcha.acousticModel);

    calloc.free(c.ref.model.vits.dataDir);
    calloc.free(c.ref.model.vits.tokens);
    calloc.free(c.ref.model.vits.lexicon);
    calloc.free(c.ref.model.vits.model);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception("Failed to create offline tts. Please check your config");
    }

    return OfflineTts._(ptr: ptr, config: config);
  }

  void free() {
    SherpaOnnxBindings.destroyOfflineTts?.call(ptr);
    ptr = nullptr;
  }

  GeneratedAudio generate(
      {required String text, int sid = 0, double speed = 1.0}) {
    final Pointer<Utf8> textPtr = text.toNativeUtf8();
    final p =
        SherpaOnnxBindings.offlineTtsGenerate?.call(ptr, textPtr, sid, speed) ??
            nullptr;
    calloc.free(textPtr);

    if (p == nullptr) {
      return GeneratedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final samples = p.ref.samples.asTypedList(p.ref.n);
    final sampleRate = p.ref.sampleRate;
    final newSamples = Float32List.fromList(samples);

    SherpaOnnxBindings.destroyOfflineTtsGeneratedAudio?.call(p);

    return GeneratedAudio(samples: newSamples, sampleRate: sampleRate);
  }

  GeneratedAudio generateWithCallback(
      {required String text,
      int sid = 0,
      double speed = 1.0,
      required int Function(Float32List samples) callback}) {
    // see
    // https://github.com/dart-lang/sdk/issues/54276#issuecomment-1846109285
    // https://stackoverflow.com/questions/69537440/callbacks-in-dart-dartffi-only-supports-calling-static-dart-functions-from-nat
    // https://github.com/dart-lang/sdk/blob/main/tests/ffi/isolate_local_function_callbacks_test.dart#L46
    final wrapper =
        NativeCallable<SherpaOnnxGeneratedAudioCallbackNative>.isolateLocal(
            (Pointer<Float> samples, int n) {
      final s = samples.asTypedList(n);
      final newSamples = Float32List.fromList(s);
      return callback(newSamples);
    }, exceptionalReturn: 0);

    final Pointer<Utf8> textPtr = text.toNativeUtf8();
    final p = SherpaOnnxBindings.offlineTtsGenerateWithCallback
            ?.call(ptr, textPtr, sid, speed, wrapper.nativeFunction) ??
        nullptr;

    calloc.free(textPtr);
    wrapper.close();

    if (p == nullptr) {
      return GeneratedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final samples = p.ref.samples.asTypedList(p.ref.n);
    final sampleRate = p.ref.sampleRate;
    final newSamples = Float32List.fromList(samples);

    SherpaOnnxBindings.destroyOfflineTtsGeneratedAudio?.call(p);

    return GeneratedAudio(samples: newSamples, sampleRate: sampleRate);
  }

  int get sampleRate => SherpaOnnxBindings.offlineTtsSampleRate?.call(ptr) ?? 0;

  int get numSpeakers =>
      SherpaOnnxBindings.offlineTtsNumSpeakers?.call(ptr) ?? 0;

  Pointer<SherpaOnnxOfflineTts> ptr;
  OfflineTtsConfig config;
}
