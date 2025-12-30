// Copyright (c)  2025  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import './sherpa_onnx_bindings.dart';

class OfflineSpeechDenoiserGtcrnModelConfig {
  const OfflineSpeechDenoiserGtcrnModelConfig({
    this.model = '',
  });

  factory OfflineSpeechDenoiserGtcrnModelConfig.fromJson(
      Map<String, dynamic> json) {
    return OfflineSpeechDenoiserGtcrnModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserGtcrnModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

class OfflineSpeechDenoiserModelConfig {
  const OfflineSpeechDenoiserModelConfig({
    this.gtcrn = const OfflineSpeechDenoiserGtcrnModelConfig(),
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
  });

  factory OfflineSpeechDenoiserModelConfig.fromJson(Map<String, dynamic> json) {
    return OfflineSpeechDenoiserModelConfig(
      gtcrn: json['gtcrn'] != null
          ? OfflineSpeechDenoiserGtcrnModelConfig.fromJson(
              json['gtcrn'] as Map<String, dynamic>)
          : const OfflineSpeechDenoiserGtcrnModelConfig(),
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserModelConfig(gtcrn: $gtcrn, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'gtcrn': gtcrn.toJson(),
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final OfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  final int numThreads;
  final bool debug;
  final String provider;
}

class OfflineSpeechDenoiserConfig {
  const OfflineSpeechDenoiserConfig({
    this.model = const OfflineSpeechDenoiserModelConfig(),
  });

  factory OfflineSpeechDenoiserConfig.fromJson(Map<String, dynamic> json) {
    return OfflineSpeechDenoiserConfig(
      model: json['model'] != null
          ? OfflineSpeechDenoiserModelConfig.fromJson(
              json['model'] as Map<String, dynamic>)
          : const OfflineSpeechDenoiserModelConfig(),
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model.toJson(),
      };

  final OfflineSpeechDenoiserModelConfig model;
}

class DenoisedAudio {
  DenoisedAudio({
    required this.samples,
    required this.sampleRate,
  });

  final Float32List samples;
  final int sampleRate;
}

class OfflineSpeechDenoiser {
  OfflineSpeechDenoiser.fromPtr({required this.ptr, required this.config});

  OfflineSpeechDenoiser._({required this.ptr, required this.config});

  /// The user is responsible to call the OfflineSpeechDenoiser.free()
  /// method of the returned instance to avoid memory leak.
  factory OfflineSpeechDenoiser(OfflineSpeechDenoiserConfig config) {
    final c = calloc<SherpaOnnxOfflineSpeechDenoiserConfig>();
    c.ref.model.gtcrn.model = config.model.gtcrn.model.toNativeUtf8();

    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.debug = config.model.debug ? 1 : 0;
    c.ref.model.provider = config.model.provider.toNativeUtf8();

    if (SherpaOnnxBindings.sherpaOnnxCreateOfflineSpeechDenoiser == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final ptr =
        SherpaOnnxBindings.sherpaOnnxCreateOfflineSpeechDenoiser?.call(c) ??
            nullptr;

    calloc.free(c.ref.model.provider);
    calloc.free(c.ref.model.gtcrn.model);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create offline speech denoiser. Please check your config");
    }

    return OfflineSpeechDenoiser._(ptr: ptr, config: config);
  }

  void free() {
    SherpaOnnxBindings.sherpaOnnxDestroyOfflineSpeechDenoiser?.call(ptr);
    ptr = nullptr;
  }

  DenoisedAudio run({required Float32List samples, required int sampleRate}) {
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

    final denoisedSamples = p.ref.samples.asTypedList(p.ref.n);
    final denoisedSampleRate = p.ref.sampleRate;
    final newSamples = Float32List.fromList(denoisedSamples);

    SherpaOnnxBindings.sherpaOnnxDestroyDenoisedAudio?.call(p);

    return DenoisedAudio(samples: newSamples, sampleRate: denoisedSampleRate);
  }

  int get sampleRate =>
      SherpaOnnxBindings.sherpaOnnxOfflineSpeechDenoiserGetSampleRate
          ?.call(ptr) ??
      0;

  Pointer<SherpaOnnxOfflineSpeechDenoiser> ptr;
  OfflineSpeechDenoiserConfig config;
}
