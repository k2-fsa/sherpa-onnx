// Copyright (c)  2026  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import './offline_speech_denoiser.dart';
import './sherpa_onnx_bindings.dart';

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

class OnlineSpeechDenoiser {
  OnlineSpeechDenoiser.fromPtr({required this.ptr, required this.config});

  OnlineSpeechDenoiser._({required this.ptr, required this.config});

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

  void reset() {
    if (SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserReset == null) {
      throw Exception('Please initialize sherpa-onnx first');
    }

    if (ptr == nullptr) {
      return;
    }

    SherpaOnnxBindings.sherpaOnnxOnlineSpeechDenoiserReset?.call(ptr);
  }

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
