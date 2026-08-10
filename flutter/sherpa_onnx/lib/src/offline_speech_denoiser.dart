// Copyright (c)  2025  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import './sherpa_onnx_bindings.dart';
import './offline_speech_denoiser_config.dart';

export './offline_speech_denoiser_config.dart';

/// Offline speech denoiser.
class OfflineSpeechDenoiser {
  OfflineSpeechDenoiser.fromPtr({required this.ptr, required this.config});

  OfflineSpeechDenoiser._({required this.ptr, required this.config});

  /// Create an offline denoiser from [config].
  factory OfflineSpeechDenoiser(OfflineSpeechDenoiserConfig config) {
    if (SherpaOnnxBindings.sherpaOnnxCreateOfflineSpeechDenoiser == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final c = calloc<SherpaOnnxOfflineSpeechDenoiserConfig>();
    c.ref.model.gtcrn.model = config.model.gtcrn.model.toNativeUtf8();
    c.ref.model.dpdfnet.model = config.model.dpdfnet.model.toNativeUtf8();
    c.ref.model.dpdfnet.attenuationLimitDb =
        config.model.dpdfnet.attenuationLimitDb;

    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.debug = config.model.debug ? 1 : 0;
    c.ref.model.provider = config.model.provider.toNativeUtf8();

    final ptr =
        SherpaOnnxBindings.sherpaOnnxCreateOfflineSpeechDenoiser?.call(c) ??
            nullptr;

    calloc.free(c.ref.model.provider);
    calloc.free(c.ref.model.gtcrn.model);
    calloc.free(c.ref.model.dpdfnet.model);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create offline speech denoiser. Please check your config");
    }

    return OfflineSpeechDenoiser._(ptr: ptr, config: config);
  }

  /// Release the native denoiser.
  void free() {
    if (SherpaOnnxBindings.sherpaOnnxDestroyOfflineSpeechDenoiser == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }

    SherpaOnnxBindings.sherpaOnnxDestroyOfflineSpeechDenoiser?.call(ptr);
    ptr = nullptr;
  }

  /// Denoise one chunk or a complete waveform.
  DenoisedAudio run({required Float32List samples, required int sampleRate}) {
    if (SherpaOnnxBindings.sherpaOnnxOfflineSpeechDenoiserRun == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return DenoisedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final n = samples.length;
    final Pointer<Float> psamples = calloc<Float>(n);

    final pList = psamples.asTypedList(n);
    pList.setAll(0, samples);

    final p = SherpaOnnxBindings.sherpaOnnxOfflineSpeechDenoiserRun
            ?.call(ptr, psamples, n, sampleRate) ??
        nullptr;

    calloc.free(psamples);

    if (p == nullptr) {
      return DenoisedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final sampleRateOut = p.ref.sampleRate;
    final nOut = p.ref.n;
    Float32List newSamples = Float32List(0);
    if (nOut > 0 && p.ref.samples != nullptr) {
      newSamples = Float32List.fromList(p.ref.samples.asTypedList(nOut));
    }

    SherpaOnnxBindings.sherpaOnnxDestroyDenoisedAudio?.call(p);

    return DenoisedAudio(samples: newSamples, sampleRate: sampleRateOut);
  }

  /// Return the expected sample rate for this denoiser.
  int get sampleRate {
    if (SherpaOnnxBindings.sherpaOnnxOfflineSpeechDenoiserGetSampleRate ==
        null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.sherpaOnnxOfflineSpeechDenoiserGetSampleRate
            ?.call(ptr) ??
        0;
  }

  Pointer<SherpaOnnxOfflineSpeechDenoiser> ptr;
  OfflineSpeechDenoiserConfig config;
}
