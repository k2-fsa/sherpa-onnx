// Copyright (c)  2025  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import './sherpa_onnx_bindings.dart';

/// Offline speech denoising.
///
/// Supported model families include GTCRN and DPDFNet. See the examples under
/// `dart-api-examples/speech-enhancement-gtcrn/` and
/// `dart-api-examples/speech-enhancement-dpdfnet/`.
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

/// DPDFNet model path for offline speech denoising.
class OfflineSpeechDenoiserDpdfNetModelConfig {
  const OfflineSpeechDenoiserDpdfNetModelConfig({
    this.model = '',
  });

  factory OfflineSpeechDenoiserDpdfNetModelConfig.fromJson(
      Map<String, dynamic> json) {
    return OfflineSpeechDenoiserDpdfNetModelConfig(
      model: json['model'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserDpdfNetModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
      };

  final String model;
}

/// Aggregate model configuration for [OfflineSpeechDenoiser].
///
/// Configure either [gtcrn] or [dpdfnet] for typical use.
class OfflineSpeechDenoiserModelConfig {
  const OfflineSpeechDenoiserModelConfig({
    this.gtcrn = const OfflineSpeechDenoiserGtcrnModelConfig(),
    this.dpdfnet = const OfflineSpeechDenoiserDpdfNetModelConfig(),
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
      dpdfnet: json['dpdfnet'] != null
          ? OfflineSpeechDenoiserDpdfNetModelConfig.fromJson(
              json['dpdfnet'] as Map<String, dynamic>)
          : const OfflineSpeechDenoiserDpdfNetModelConfig(),
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeechDenoiserModelConfig(gtcrn: $gtcrn, dpdfnet: $dpdfnet, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'gtcrn': gtcrn.toJson(),
        'dpdfnet': dpdfnet.toJson(),
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final OfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  final OfflineSpeechDenoiserDpdfNetModelConfig dpdfnet;
  final int numThreads;
  final bool debug;
  final String provider;
}

/// Top-level configuration for [OfflineSpeechDenoiser].
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

/// Audio returned by offline or online speech denoisers.
class DenoisedAudio {
  DenoisedAudio({
    required this.samples,
    required this.sampleRate,
  });

  final Float32List samples;
  final int sampleRate;
}

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
