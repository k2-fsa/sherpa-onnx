// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './offline_stream.dart';
import './sherpa_onnx_bindings.dart';
import './utils.dart';
import './spoken_language_identification_config.dart';

export './spoken_language_identification_config.dart';

/// Spoken language identifier.
class SpokenLanguageIdentification {
  SpokenLanguageIdentification.fromPtr(
      {required this.ptr, required this.config});

  SpokenLanguageIdentification._({required this.ptr, required this.config});

  /// Release the native language identifier.
  void free() {
    if (SherpaOnnxBindings.sherpaOnnxDestroySpokenLanguageIdentification ==
        null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.sherpaOnnxDestroySpokenLanguageIdentification?.call(ptr);
    ptr = nullptr;
  }

  /// Create a language identifier from [config].
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

  /// Create an offline stream for one audio clip.
  OfflineStream createStream() {
    if (SherpaOnnxBindings
            .sherpaOnnxSpokenLanguageIdentificationCreateOfflineStream ==
        null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      throw Exception("Failed to create offline stream");
    }

    final p = SherpaOnnxBindings
            .sherpaOnnxSpokenLanguageIdentificationCreateOfflineStream
            ?.call(ptr) ??
        nullptr;

    if (p == nullptr) {
      throw Exception("Failed to create offline stream");
    }

    return OfflineStream(ptr: p);
  }

  /// Compute the spoken language for [stream].
  SpokenLanguageIdentificationResult compute(OfflineStream stream) {
    if (SherpaOnnxBindings.sherpaOnnxSpokenLanguageIdentificationCompute ==
        null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return const SpokenLanguageIdentificationResult(lang: '');
    }

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
