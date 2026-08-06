// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './offline_stream.dart';
import './sherpa_onnx_bindings.dart';
import './utils.dart';
import './offline_recognizer_config.dart';

export './offline_recognizer_config.dart';

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
