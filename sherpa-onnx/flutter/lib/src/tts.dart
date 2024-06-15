// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

class OfflineTtsVitsModelConfig {
  const OfflineTtsVitsModelConfig({
    required this.model,
    this.lexicon = '',
    required this.tokens,
    this.dataDir = '',
    this.noiseScale = 0.667,
    this.noiseScaleW = 0.8,
    this.lengthScale = 1.0,
    this.dictDir = '',
  });

  @override
  String toString() {
    return 'OfflineTtsVitsModelConfig(model: $model, lexicon: $lexicon, tokens: $tokens, dataDir: $dataDir, noiseScale: $noiseScale, noiseScaleW: $noiseScaleW, lengthScale: $lengthScale, dictDir: $dictDir)';
  }

  final String model;
  final String lexicon;
  final String tokens;
  final String dataDir;
  final double noiseScale;
  final double noiseScaleW;
  final double lengthScale;
  final String dictDir;
}

class OfflineTtsModelConfig {
  const OfflineTtsModelConfig({
    required this.vits,
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
  });

  @override
  String toString() {
    return 'OfflineTtsModelConfig(vits: $vits, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  final OfflineTtsVitsModelConfig vits;
  final int numThreads;
  final bool debug;
  final String provider;
}

class OfflineTtsConfig {
  const OfflineTtsConfig({
    required this.model,
    this.ruleFsts = '',
    this.maxNumSenetences = 1,
    this.ruleFars = '',
  });

  @override
  String toString() {
    return 'OfflineTtsConfig(model: $model, ruleFsts: $ruleFsts, maxNumSenetences: $maxNumSenetences, ruleFars: $ruleFars)';
  }

  final OfflineTtsModelConfig model;
  final String ruleFsts;
  final int maxNumSenetences;
  final String ruleFars;
}

class GeneratedAudio {
  GeneratedAudio({
    required this.samples,
    required this.sampleRate,
  });

  final Float32List samples;
  final int sampleRate;
}

class OfflineTts {
  OfflineTts._({required this.ptr, required this.config});

  /// The user is responsible to call the OfflineTts.free()
  /// method of the returned instance to avoid memory leak.
  factory OfflineTts(OfflineTtsConfig config) {
    final c = calloc<SherpaOnnxOfflineTtsConfig>();
    c.ref.model.vits.model = config.model.vits.model.toNativeUtf8();
    c.ref.model.vits.lexicon = config.model.vits.lexicon.toNativeUtf8();
    c.ref.model.vits.tokens = config.model.vits.tokens.toNativeUtf8();
    c.ref.model.vits.dataDir = config.model.vits.dataDir.toNativeUtf8();
    c.ref.model.vits.noiseScale = config.model.vits.noiseScale;
    c.ref.model.vits.noiseScaleW = config.model.vits.noiseScaleW;
    c.ref.model.vits.lengthScale = config.model.vits.lengthScale;
    c.ref.model.vits.dictDir = config.model.vits.dictDir.toNativeUtf8();

    c.ref.model.numThreads = config.model.numThreads;
    c.ref.model.debug = config.model.debug ? 1 : 0;
    c.ref.model.provider = config.model.provider.toNativeUtf8();

    c.ref.ruleFsts = config.ruleFsts.toNativeUtf8();
    c.ref.maxNumSenetences = config.maxNumSenetences;
    c.ref.ruleFars = config.ruleFars.toNativeUtf8();

    final ptr = SherpaOnnxBindings.createOfflineTts?.call(c) ?? nullptr;

    calloc.free(c.ref.ruleFars);
    calloc.free(c.ref.ruleFsts);
    calloc.free(c.ref.model.provider);
    calloc.free(c.ref.model.vits.dictDir);
    calloc.free(c.ref.model.vits.dataDir);
    calloc.free(c.ref.model.vits.tokens);
    calloc.free(c.ref.model.vits.lexicon);
    calloc.free(c.ref.model.vits.model);

    return OfflineTts._(ptr: ptr, config: config);
  }

  void free() {
    SherpaOnnxBindings.destroyOfflineTts?.call(ptr);
    ptr = nullptr;
  }

  GeneratedAudio generate(
      {required String text, int sid = 0, double speed = 1.0}) {
    final Pointer<Utf8> textPtr = text.toNativeUtf8();
    final p =
        SherpaOnnxBindings.offlineTtsGenerate?.call(ptr, textPtr, sid, speed) ??
            nullptr;
    calloc.free(textPtr);

    if (p == nullptr) {
      return GeneratedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final samples = p.ref.samples.asTypedList(p.ref.n);
    final sampleRate = p.ref.sampleRate;
    final newSamples = Float32List.fromList(samples);

    SherpaOnnxBindings.destroyOfflineTtsGeneratedAudio?.call(p);

    return GeneratedAudio(samples: newSamples, sampleRate: sampleRate);
  }

  GeneratedAudio generateWithCallback(
      {required String text,
      int sid = 0,
      double speed = 1.0,
      required void Function(Float32List samples) callback}) {
    // see
    // https://github.com/dart-lang/sdk/issues/54276#issuecomment-1846109285
    // https://stackoverflow.com/questions/69537440/callbacks-in-dart-dartffi-only-supports-calling-static-dart-functions-from-nat
    // https://github.com/dart-lang/sdk/blob/main/tests/ffi/isolate_local_function_callbacks_test.dart#L46
    final wrapper =
        NativeCallable<SherpaOnnxGeneratedAudioCallbackNative>.isolateLocal(
            (Pointer<Float> samples, int n) {
      final s = samples.asTypedList(n);
      final newSamples = Float32List.fromList(s);
      callback(newSamples);
    });

    final Pointer<Utf8> textPtr = text.toNativeUtf8();
    final p = SherpaOnnxBindings.offlineTtsGenerateWithCallback
            ?.call(ptr, textPtr, sid, speed, wrapper.nativeFunction) ??
        nullptr;

    calloc.free(textPtr);
    wrapper.close();

    if (p == nullptr) {
      return GeneratedAudio(samples: Float32List(0), sampleRate: 0);
    }

    final samples = p.ref.samples.asTypedList(p.ref.n);
    final sampleRate = p.ref.sampleRate;
    final newSamples = Float32List.fromList(samples);

    SherpaOnnxBindings.destroyOfflineTtsGeneratedAudio?.call(p);

    return GeneratedAudio(samples: newSamples, sampleRate: sampleRate);
  }

  int get sampleRate =>
      SherpaOnnxBindings.offlineTtsSampleRate?.call(this.ptr) ?? 0;

  int get numSpeakers =>
      SherpaOnnxBindings.offlineTtsNumSpeakers?.call(this.ptr) ?? 0;

  Pointer<SherpaOnnxOfflineTts> ptr;
  OfflineTtsConfig config;
}
