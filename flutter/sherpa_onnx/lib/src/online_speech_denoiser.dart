// Copyright (c)  2026  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import './offline_speech_denoiser.dart';
import './sherpa_onnx_bindings.dart';

/// Streaming speech denoising.
///
/// Call [run] on consecutive chunks, then [flush] after the final chunk to
/// drain any buffered state.
class OnlineSpeechDenoiserConfig {
  const OnlineSpeechDenoiserConfig({
    this.model = const OfflineSpeechDenoiserModelConfig(),
  });

  factory OnlineSpeechDenoiserConfig.fromJson(Map<String, dynamic> json) {
    return OnlineSpeechDenoiserConfig(
      model: json['model'] != null
          ? OfflineSpeechDenoiserModelConfig.fromJson(
              json['model'] as Map<String, dynamic>,
            )
          : const OfflineSpeechDenoiserModelConfig(),
    );
  }

  @override
  String toString() {
    return 'OnlineSpeechDenoiserConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model.toJson(),
      };

  final OfflineSpeechDenoiserModelConfig model;
}

/// Streaming speech denoiser.
class OnlineSpeechDenoiser {
  OnlineSpeechDenoiser.fromPtr({required this.ptr, required this.config});

  OnlineSpeechDenoiser._({required this.ptr, required this.config});

  /// Create a streaming denoiser from [config].
  factory OnlineSpeechDenoiser(OnlineSpeechDenoiserConfig config) {
    if (SherpaOnnxBindings.sherpaOnnxCreateOnlineSpeechDenoiser == null) {
      throw Exception('Please initialize sherpa-onnx first');
    }

    final c = calloc<SherpaOnnxOnlineSpeechDenoiserConfig>();
    c.ref.model.gtcrn.model = config.model.gtcrn.model.toNativeUtf8();
    c.ref.model.dpdfnet.model = config.model.dpdfnet.model.toNativeUtf8();
    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.debug = config.model.debug ? 1 : 0;
    c.ref.model.provider = config.model.provider.toNativeUtf8();

    final ptr =
        SherpaOnnxBindings.sherpaOnnxCreateOnlineSpeechDenoiser?.call(c) ??
            nullptr;

    calloc.free(c.ref.model.provider);
    calloc.free(c.ref.model.gtcrn.model);
    calloc.free(c.ref.model.dpdfnet.model);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
        'Failed to create online speech denoiser. Please check your config',
      );
    }

    return OnlineSpeechDenoiser._(ptr: ptr, config: config);
  }

  /// Release the native denoiser.
  void free() {
    if (SherpaOnnxBindings.sherpaOnnxDestroyOnlineSpeechDenoiser == null) {
      throw Exception('Please initialize sherpa-onnx first');
    }

    if (ptr == nullptr) {
      return;
    }

    SherpaOnnxBindings.sherpaOnnxDestroyOnlineSpeechDenoiser?.call(ptr);
    ptr = nullptr;
  }

  /// Denoise one input chunk.
  DenoisedAudio run({required Float32List samples, required int sampleRate}) {
    if (SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserRun == null) {
      throw Exception('Please initialize sherpa-onnx first');
    }

    if (ptr == nullptr) {
      return DenoisedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final n = samples.length;
    final Pointer<Float> psamples = calloc<Float>(n);
    final pList = psamples.asTypedList(n);
    pList.setAll(0, samples);

    final p =
        SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserRun?.call(
              ptr,
              psamples,
              n,
              sampleRate,
            ) ??
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

  /// Flush buffered output after the final chunk.
  DenoisedAudio flush() {
    if (SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserFlush == null) {
      throw Exception('Please initialize sherpa-onnx first');
    }

    if (ptr == nullptr) {
      return DenoisedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final p =
        SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserFlush?.call(ptr) ??
            nullptr;

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

  /// Reset the streaming state.
  void reset() {
    if (SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserReset == null) {
      throw Exception('Please initialize sherpa-onnx first');
    }

    if (ptr == nullptr) {
      return;
    }

    SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserReset?.call(ptr);
  }

  /// Return the expected sample rate for this denoiser.
  int get sampleRate {
    if (SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserGetSampleRate ==
        null) {
      throw Exception('Please initialize sherpa-onnx first');
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserGetSampleRate?.call(
          ptr,
        ) ??
        0;
  }

  /// Return the preferred frame shift in samples.
  int get frameShiftInSamples {
    if (SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples ==
        null) {
      throw Exception('Please initialize sherpa-onnx first');
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples
            ?.call(ptr) ??
        0;
  }

  Pointer<SherpaOnnxOnlineSpeechDenoiser> ptr;
  OnlineSpeechDenoiserConfig config;
}
