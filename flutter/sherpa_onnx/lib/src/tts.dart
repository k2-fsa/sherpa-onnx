// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';
import './tts_config.dart';

export './tts_config.dart';

/// Offline text-to-speech.
///
/// This module supports VITS, Matcha, Kokoro, Kitten, ZipVoice, Pocket TTS,
/// and Supertonic model families. See `dart-api-examples/tts/bin/` for working
/// examples such as `pocket-en.dart`, `kokoro-en.dart`, `kokoro-zh-en.dart`,
/// `matcha-en.dart`, and `zipvoice-zh-en.dart`.
///
/// Example:
///
/// ```dart
/// final model = OfflineTtsModelConfig(
///   pocketTts: const OfflineTtsPocketSphinxModelConfig(
///     model: './sherpa-onnx-pocket-tts/model.int8.onnx',
///     tokens: './sherpa-onnx-pocket-tts/tokens.txt',
///     dataDir: './sherpa-onnx-pocket-tts/espeak-ng-data',
///   ),
///   numThreads: 1,
/// );
///
/// final tts = OfflineTts(OfflineTtsConfig(model: model));
/// final audio = tts.generate(
///   text: 'Hello from sherpa-onnx',
///   sid: 0,
///   speed: 1.0,
/// );
/// writeWave(
///   filename: './out.wav',
///   samples: audio.samples,
///   sampleRate: audio.sampleRate,
/// );
/// tts.free();
/// ```

/// FFI bridge methods for [OfflineTtsGenerationConfig].
extension OfflineTtsGenerationConfigFfi on OfflineTtsGenerationConfig {
  /// Convert Extra to JSON string.
  /// Returns nullptr if empty.
  /// The user should use calloc.free(p); to free the returned value
  Pointer<Utf8> extraToNativeUtf8() {
    if (extra.isEmpty) {
      return nullptr;
    }

    // Validate values
    for (final v in extra.values) {
      if (v is! String && v is! int && v is! double) {
        throw ArgumentError(
          'extra values must be String, int, or double. Got: ${v.runtimeType}',
        );
      }
    }

    final jsonString = jsonEncode(extra);
    return jsonString.toNativeUtf8();
  }

  Pointer<SherpaOnnxGenerationConfig> toNative() {
    final p = calloc<SherpaOnnxGenerationConfig>();

    p.ref.silenceScale = silenceScale;
    p.ref.speed = speed;
    p.ref.sid = sid;
    p.ref.numSteps = numSteps;

    if (referenceAudio != null && referenceAudio!.isNotEmpty) {
      final audioPtr = calloc<Float>(referenceAudio!.length);
      audioPtr.asTypedList(referenceAudio!.length).setAll(0, referenceAudio!);
      p.ref.referenceAudio = audioPtr;
      p.ref.referenceAudioLength = referenceAudio!.length;
      p.ref.referenceSampleRate = referenceSampleRate;
    } else {
      p.ref.referenceAudio = nullptr;
      p.ref.referenceAudioLength = 0;
      p.ref.referenceSampleRate = 0;
    }

    p.ref.referenceText = referenceText.isEmpty
        ? nullptr
        : referenceText.toNativeUtf8();

    p.ref.extra = extraToNativeUtf8();

    return p;
  }

  void freeNative(Pointer<SherpaOnnxGenerationConfig> p) {
    if (p.ref.referenceAudio != nullptr) {
      calloc.free(p.ref.referenceAudio);
    }
    if (p.ref.referenceText != nullptr) {
      calloc.free(p.ref.referenceText);
    }
    if (p.ref.extra != nullptr) {
      calloc.free(p.ref.extra);
    }
    calloc.free(p);
  }
}

/// Offline text-to-speech engine.
///
/// Create one from an [OfflineTtsConfig], then call [generate],
/// [generateWithCallback], or [generateWithConfig] depending on how much
/// control you need over the generation process.
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

    c.ref.model.matcha.acousticModel = config.model.matcha.acousticModel
        .toNativeUtf8();
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
    c.ref.model.zipvoice.encoder = config.model.zipvoice.encoder.toNativeUtf8();
    c.ref.model.zipvoice.decoder = config.model.zipvoice.decoder.toNativeUtf8();
    c.ref.model.zipvoice.vocoder = config.model.zipvoice.vocoder.toNativeUtf8();
    c.ref.model.zipvoice.dataDir = config.model.zipvoice.dataDir.toNativeUtf8();
    c.ref.model.zipvoice.lexicon = config.model.zipvoice.lexicon.toNativeUtf8();
    c.ref.model.zipvoice.featScale = config.model.zipvoice.featScale;
    c.ref.model.zipvoice.tShift = config.model.zipvoice.tShift;
    c.ref.model.zipvoice.targetRms = config.model.zipvoice.targetRms;
    c.ref.model.zipvoice.guidanceScale = config.model.zipvoice.guidanceScale;

    c.ref.model.pocket.lmFlow = config.model.pocket.lmFlow.toNativeUtf8();
    c.ref.model.pocket.lmMain = config.model.pocket.lmMain.toNativeUtf8();
    c.ref.model.pocket.encoder = config.model.pocket.encoder.toNativeUtf8();
    c.ref.model.pocket.decoder = config.model.pocket.decoder.toNativeUtf8();
    c.ref.model.pocket.textConditioner = config.model.pocket.textConditioner
        .toNativeUtf8();
    c.ref.model.pocket.vocabJson = config.model.pocket.vocabJson.toNativeUtf8();
    c.ref.model.pocket.tokenScoresJson = config.model.pocket.tokenScoresJson
        .toNativeUtf8();
    c.ref.model.pocket.voiceEmbeddingCacheCapacity =
        config.model.pocket.voiceEmbeddingCacheCapacity;

    c.ref.model.supertonic.durationPredictor = config.model.supertonic
        .durationPredictor.toNativeUtf8();
    c.ref.model.supertonic.textEncoder = config.model.supertonic.textEncoder
        .toNativeUtf8();
    c.ref.model.supertonic.vectorEstimator = config.model.supertonic
        .vectorEstimator.toNativeUtf8();
    c.ref.model.supertonic.vocoder = config.model.supertonic.vocoder
        .toNativeUtf8();
    c.ref.model.supertonic.ttsJson = config.model.supertonic.ttsJson
        .toNativeUtf8();
    c.ref.model.supertonic.unicodeIndexer = config.model.supertonic
        .unicodeIndexer.toNativeUtf8();
    c.ref.model.supertonic.voiceStyle = config.model.supertonic.voiceStyle
        .toNativeUtf8();

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

    calloc.free(c.ref.model.supertonic.voiceStyle);
    calloc.free(c.ref.model.supertonic.unicodeIndexer);
    calloc.free(c.ref.model.supertonic.ttsJson);
    calloc.free(c.ref.model.supertonic.vocoder);
    calloc.free(c.ref.model.supertonic.vectorEstimator);
    calloc.free(c.ref.model.supertonic.textEncoder);
    calloc.free(c.ref.model.supertonic.durationPredictor);

    calloc.free(c.ref.model.pocket.tokenScoresJson);
    calloc.free(c.ref.model.pocket.vocabJson);
    calloc.free(c.ref.model.pocket.textConditioner);
    calloc.free(c.ref.model.pocket.decoder);
    calloc.free(c.ref.model.pocket.encoder);
    calloc.free(c.ref.model.pocket.lmMain);
    calloc.free(c.ref.model.pocket.lmFlow);

    calloc.free(c.ref.model.zipvoice.lexicon);
    calloc.free(c.ref.model.zipvoice.dataDir);
    calloc.free(c.ref.model.zipvoice.vocoder);
    calloc.free(c.ref.model.zipvoice.decoder);
    calloc.free(c.ref.model.zipvoice.encoder);
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

  /// Release the native TTS engine.
  void free() {
    if (SherpaOnnxBindings.destroyOfflineTts == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroyOfflineTts?.call(ptr);
    ptr = nullptr;
  }

  /// Generate audio using the simple `(text, sid, speed)` API.
  GeneratedAudio generate({
    required String text,
    int sid = 0,
    double speed = 1.0,
  }) {
    return generateWithConfig(
      text: text,
      config: OfflineTtsGenerationConfig(sid: sid, speed: speed),
    );
  }

  /// Generate audio while receiving partial sample chunks through [callback].
  GeneratedAudio generateWithCallback({
    required String text,
    int sid = 0,
    double speed = 1.0,
    required int Function(Float32List samples) callback,
  }) {
    return generateWithConfig(
      text: text,
      config: OfflineTtsGenerationConfig(sid: sid, speed: speed),
      onProgress: (samples, progress) => callback(samples),
    );
  }

  /// Generate audio using [OfflineTtsGenerationConfig].
  ///
  /// This is the most flexible generation API and is the recommended entry
  /// point for features such as Pocket TTS reference-audio cloning and
  /// model-specific options supplied through [OfflineTtsGenerationConfig.extra].
  GeneratedAudio generateWithConfig({
    required String text,
    required OfflineTtsGenerationConfig config,
    int Function(Float32List samples, double progress)? onProgress,
  }) {
    if (SherpaOnnxBindings.offlineTtsGenerateWithConfig == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return GeneratedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final textPtr = text.toNativeUtf8();
    final cfgPtr = config.toNative();

    NativeCallable<SherpaOnnxGeneratedAudioProgressCallbackWithArgNative>?
    wrapper;

    if (onProgress != null) {
      wrapper =
          NativeCallable<
            SherpaOnnxGeneratedAudioProgressCallbackWithArgNative
          >.isolateLocal((
            Pointer<Float> samples,
            int n,
            double p,
            Pointer<Void> arg,
          ) {
            final list = Float32List.fromList(samples.asTypedList(n));
            return onProgress(list, p);
          }, exceptionalReturn: 0);
    }

    final p =
        SherpaOnnxBindings.offlineTtsGenerateWithConfig?.call(
          ptr,
          textPtr,
          cfgPtr,
          wrapper?.nativeFunction ?? nullptr,
          nullptr,
        ) ??
        nullptr;

    calloc.free(textPtr);
    config.freeNative(cfgPtr);
    wrapper?.close();

    if (p == nullptr) {
      return GeneratedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final samples = Float32List.fromList(p.ref.samples.asTypedList(p.ref.n));
    final sampleRate = p.ref.sampleRate;

    SherpaOnnxBindings.destroyOfflineTtsGeneratedAudio?.call(p);

    return GeneratedAudio(samples: samples, sampleRate: sampleRate);
  }

  /// Return the output sample rate reported by the model.
  int get sampleRate {
    if (SherpaOnnxBindings.offlineTtsSampleRate == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.offlineTtsSampleRate?.call(ptr) ?? 0;
  }

  /// Return the number of built-in speakers reported by the model.
  int get numSpeakers {
    if (SherpaOnnxBindings.offlineTtsNumSpeakers == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.offlineTtsNumSpeakers?.call(ptr) ?? 0;
  }

  Pointer<SherpaOnnxOfflineTts> ptr;
  OfflineTtsConfig config;
}
