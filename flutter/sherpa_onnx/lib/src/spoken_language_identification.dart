// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './offline_stream.dart';
import './sherpa_onnx_bindings.dart';
import './utils.dart';

class SpokenLanguageIdentificationWhisperConfig {
  const SpokenLanguageIdentificationWhisperConfig({
    this.encoder = '',
    this.decoder = '',
    this.tailPaddings = 0,
  });

  factory SpokenLanguageIdentificationWhisperConfig.fromJson(
      Map<String, dynamic> json) {
    return SpokenLanguageIdentificationWhisperConfig(
      encoder: json['encoder'] as String? ?? '',
      decoder: json['decoder'] as String? ?? '',
      tailPaddings: json['tailPaddings'] as int? ?? 0,
    );
  }

  @override
  String toString() {
    return 'SpokenLanguageIdentificationWhisperConfig(encoder: $encoder, decoder: $decoder, tailPaddings: $tailPaddings)';
  }

  Map<String, dynamic> toJson() => {
        'encoder': encoder,
        'decoder': decoder,
        'tailPaddings': tailPaddings,
      };

  final String encoder;
  final String decoder;
  final int tailPaddings;
}

class SpokenLanguageIdentificationConfig {
  const SpokenLanguageIdentificationConfig({
    this.whisper = const SpokenLanguageIdentificationWhisperConfig(),
    this.numThreads = 1,
    this.debug = false,
    this.provider = 'cpu',
  });

  factory SpokenLanguageIdentificationConfig.fromJson(
      Map<String, dynamic> json) {
    return SpokenLanguageIdentificationConfig(
      whisper: json['whisper'] != null
          ? SpokenLanguageIdentificationWhisperConfig.fromJson(
              json['whisper'] as Map<String, dynamic>)
          : const SpokenLanguageIdentificationWhisperConfig(),
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? false,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'SpokenLanguageIdentificationConfig(whisper: $whisper, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'whisper': whisper.toJson(),
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final SpokenLanguageIdentificationWhisperConfig whisper;
  final int numThreads;
  final bool debug;
  final String provider;
}

class SpokenLanguageIdentificationResult {
  const SpokenLanguageIdentificationResult({
    required this.lang,
  });

  factory SpokenLanguageIdentificationResult.fromJson(
      Map<String, dynamic> json) {
    return SpokenLanguageIdentificationResult(
      lang: json['lang'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'SpokenLanguageIdentificationResult(lang: $lang)';
  }

  Map<String, dynamic> toJson() => {
        'lang': lang,
      };

  final String lang;
}

class SpokenLanguageIdentification {
  SpokenLanguageIdentification.fromPtr(
      {required this.ptr, required this.config});

  SpokenLanguageIdentification._({required this.ptr, required this.config});

  void free() {
    SherpaOnnxBindings.sherpaOnnxDestroySpokenLanguageIdentification?.call(ptr);
    ptr = nullptr;
  }

  /// The user is responsible to call the SpokenLanguageIdentification.free()
  /// method of the returned instance to avoid memory leak.
  factory SpokenLanguageIdentification(
      SpokenLanguageIdentificationConfig config) {
    final c = convertConfig(config);

    if (SherpaOnnxBindings.sherpaOnnxCreateSpokenLanguageIdentification ==
        null) {
      freeConfig(c);
      throw Exception("Please initialize sherpa-onnx first");
    }

    final ptr = SherpaOnnxBindings.sherpaOnnxCreateSpokenLanguageIdentification
            ?.call(c) ??
        nullptr;

    if (ptr == nullptr) {
      freeConfig(c);
      throw Exception(
          "Failed to create spoken language identification. Please check your config");
    }

    freeConfig(c);

    return SpokenLanguageIdentification._(ptr: ptr, config: config);
  }

  static Pointer<SherpaOnnxSpokenLanguageIdentificationConfig> convertConfig(
      SpokenLanguageIdentificationConfig config) {
    final c = calloc<SherpaOnnxSpokenLanguageIdentificationConfig>();

    c.ref.whisper.encoder = config.whisper.encoder.toNativeUtf8();
    c.ref.whisper.decoder = config.whisper.decoder.toNativeUtf8();
    c.ref.whisper.tailPaddings = config.whisper.tailPaddings;

    c.ref.numThreads = config.numThreads;
    c.ref.debug = config.debug ? 1 : 0;
    c.ref.provider = config.provider.toNativeUtf8();

    return c;
  }

  static void freeConfig(
      Pointer<SherpaOnnxSpokenLanguageIdentificationConfig> c) {
    malloc.free(c.ref.whisper.encoder);
    malloc.free(c.ref.whisper.decoder);
    malloc.free(c.ref.provider);
    malloc.free(c);
  }

  /// The user has to invoke stream.free() on the returned instance
  /// to avoid memory leak
  OfflineStream createStream() {
    final p = SherpaOnnxBindings
            .sherpaOnnxSpokenLanguageIdentificationCreateOfflineStream
            ?.call(ptr) ??
        nullptr;
    return OfflineStream(ptr: p);
  }

  SpokenLanguageIdentificationResult compute(OfflineStream stream) {
    final result = SherpaOnnxBindings
            .sherpaOnnxSpokenLanguageIdentificationCompute
            ?.call(ptr, stream.ptr) ??
        nullptr;

    if (result == nullptr) {
      return const SpokenLanguageIdentificationResult(lang: '');
    }

    final lang = toDartString(result.ref.lang);

    SherpaOnnxBindings.sherpaOnnxDestroySpokenLanguageIdentificationResult
        ?.call(result);

    return SpokenLanguageIdentificationResult(lang: lang);
  }

  Pointer<SherpaOnnxSpokenLanguageIdentification> ptr;
  SpokenLanguageIdentificationConfig config;
}
