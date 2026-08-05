// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './online_recognizer_config.dart';
import './online_stream.dart';
import './sherpa_onnx_bindings.dart';
import './utils.dart';

export './online_recognizer_config.dart';

/// Streaming speech recognition.
///
/// This module wraps the online ASR APIs used by the examples in
/// `dart-api-examples/streaming-asr/bin/`, including Zipformer transducer,
/// Zipformer CTC, Paraformer, T-One-CTC, and NeMo-CTC style models.
///
/// Example:
///
/// ```dart
/// final model = OnlineModelConfig(
///   transducer: const OnlineTransducerModelConfig(
///     encoder: './streaming-zipformer/encoder-epoch-99-avg-1.int8.onnx',
///     decoder: './streaming-zipformer/decoder-epoch-99-avg-1.onnx',
///     joiner: './streaming-zipformer/joiner-epoch-99-avg-1.int8.onnx',
///   ),
///   tokens: './streaming-zipformer/tokens.txt',
///   modelType: 'zipformer2',
/// );
///
/// final recognizer = OnlineRecognizer(OnlineRecognizerConfig(model: model));
/// final stream = recognizer.createStream();
/// stream.acceptWaveform(samples: chunk, sampleRate: 16000);
/// while (recognizer.isReady(stream)) {
///   recognizer.decode(stream);
/// }
/// print(recognizer.getResult(stream).text);
/// ```

/// Streaming speech recognizer.
///
/// Create one from an [OnlineRecognizerConfig], then feed chunks to an
/// [OnlineStream] and call [decode] while [isReady] is true.
class OnlineRecognizer {
  OnlineRecognizer.fromPtr({required this.ptr, required this.config});

  OnlineRecognizer._({required this.ptr, required this.config});

  /// The user is responsible to call the OnlineRecognizer.free()
  /// method of the returned instance to avoid memory leak.
  /// Create a recognizer from [config].
  factory OnlineRecognizer(OnlineRecognizerConfig config) {
    if (SherpaOnnxBindings.createOnlineRecognizer == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

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

    // nemoCtc
    c.ref.model.nemoCtc.model = config.model.nemoCtc.model.toNativeUtf8();

    // toneCtc
    c.ref.model.toneCtc.model = config.model.toneCtc.model.toNativeUtf8();

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

    c.ref.blankPenalty = config.blankPenalty;

    c.ref.hr.lexicon = config.hr.lexicon.toNativeUtf8();
    c.ref.hr.ruleFsts = config.hr.ruleFsts.toNativeUtf8();

    final ptr = SherpaOnnxBindings.createOnlineRecognizer?.call(c) ?? nullptr;

    calloc.free(c.ref.hr.lexicon);
    calloc.free(c.ref.hr.ruleFsts);
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
    calloc.free(c.ref.model.toneCtc.model);
    calloc.free(c.ref.model.nemoCtc.model);
    calloc.free(c.ref.model.zipformer2Ctc.model);
    calloc.free(c.ref.model.paraformer.encoder);
    calloc.free(c.ref.model.paraformer.decoder);

    calloc.free(c.ref.model.transducer.encoder);
    calloc.free(c.ref.model.transducer.decoder);
    calloc.free(c.ref.model.transducer.joiner);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create online recognizer. Please check your config");
    }

    return OnlineRecognizer._(ptr: ptr, config: config);
  }

  /// Release the native recognizer.
  void free() {
    if (SherpaOnnxBindings.destroyOnlineRecognizer == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroyOnlineRecognizer?.call(ptr);
    ptr = nullptr;
  }

  /// The user has to invoke stream.free() on the returned instance
  /// to avoid memory leak
  /// Create a streaming input stream.
  ///
  /// If [hotwords] is provided, the stream uses those per-stream hotwords in
  /// addition to any recognizer-wide settings.
  OnlineStream createStream({String hotwords = ''}) {
    if (hotwords == '') {
      if (SherpaOnnxBindings.createOnlineStream == null) {
        throw Exception("Please initialize sherpa-onnx first");
      }
    } else {
      if (SherpaOnnxBindings.createOnlineStreamWithHotwords == null) {
        throw Exception("Please initialize sherpa-onnx first");
      }
    }

    if (ptr == nullptr) {
      throw Exception("Failed to create online stream");
    }

    if (hotwords == '') {
      final p = SherpaOnnxBindings.createOnlineStream?.call(ptr) ?? nullptr;
      if (p == nullptr) {
        throw Exception("Failed to create online stream");
      }
      return OnlineStream(ptr: p);
    }

    final utf8 = hotwords.toNativeUtf8();
    final p =
        SherpaOnnxBindings.createOnlineStreamWithHotwords?.call(ptr, utf8) ??
            nullptr;
    calloc.free(utf8);

    if (p == nullptr) {
      throw Exception("Failed to create online stream");
    }

    return OnlineStream(ptr: p);
  }

  /// Return `true` if the recognizer has enough audio to run another step.
  bool isReady(OnlineStream stream) {
    if (SherpaOnnxBindings.isOnlineStreamReady == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return false;
    }

    int ready =
        SherpaOnnxBindings.isOnlineStreamReady?.call(ptr, stream.ptr) ?? 0;

    return ready == 1;
  }

  /// Fetch the current recognition hypothesis.
  OnlineRecognizerResult getResult(OnlineStream stream) {
    if (SherpaOnnxBindings.getOnlineStreamResultAsJson == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return OnlineRecognizerResult(text: '', tokens: [], timestamps: []);
    }

    final json =
        SherpaOnnxBindings.getOnlineStreamResultAsJson?.call(ptr, stream.ptr) ??
            nullptr;
    if (json == nullptr) {
      return OnlineRecognizerResult(text: '', tokens: [], timestamps: []);
    }

    final parsedJson = jsonDecode(toDartString(json));

    SherpaOnnxBindings.destroyOnlineStreamResultJson?.call(json);

    return OnlineRecognizerResult(
        text: parsedJson['text'],
        tokens: List<String>.from(parsedJson['tokens']),
        timestamps: List<double>.from(parsedJson['timestamps']));
  }

  /// Reset stream state after an endpoint or utterance boundary.
  void reset(OnlineStream stream) {
    if (SherpaOnnxBindings.reset == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return;
    }

    SherpaOnnxBindings.reset?.call(ptr, stream.ptr);
  }

  /// Decode one incremental step for [stream].
  void decode(OnlineStream stream) {
    if (SherpaOnnxBindings.decodeOnlineStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return;
    }

    SherpaOnnxBindings.decodeOnlineStream?.call(ptr, stream.ptr);
  }

  /// Return `true` if endpointing rules say the current utterance has ended.
  bool isEndpoint(OnlineStream stream) {
    if (SherpaOnnxBindings.isEndpoint == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return false;
    }

    int yes = SherpaOnnxBindings.isEndpoint?.call(ptr, stream.ptr) ?? 0;

    return yes == 1;
  }

  Pointer<SherpaOnnxOnlineRecognizer> ptr;
  OnlineRecognizerConfig config;
}
