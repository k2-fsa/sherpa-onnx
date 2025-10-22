// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';

import './offline_stream.dart';
import './sherpa_onnx_bindings.dart';

class OfflineZipformerAudioTaggingModelConfig {
  const OfflineZipformerAudioTaggingModelConfig({this.model = ''});

  factory OfflineZipformerAudioTaggingModelConfig.fromJson(
      Map<String, dynamic> map) {
    return OfflineZipformerAudioTaggingModelConfig(
      model: map['model'] ?? '',
    );
  }

  @override
  String toString() {
    return 'OfflineZipformerAudioTaggingModelConfig(model: $model)';
  }

  Map<String, dynamic> toJson() {
    return {
      'model': model,
    };
  }

  final String model;
}

class AudioTaggingModelConfig {
  AudioTaggingModelConfig(
      {this.zipformer = const OfflineZipformerAudioTaggingModelConfig(),
      this.ced = '',
      this.numThreads = 1,
      this.provider = 'cpu',
      this.debug = true});

  factory AudioTaggingModelConfig.fromJson(Map<String, dynamic> map) {
    return AudioTaggingModelConfig(
      zipformer:
          OfflineZipformerAudioTaggingModelConfig.fromJson(map['zipformer']),
      ced: map['ced'] ?? '',
      numThreads: map['numThreads'] ?? 1,
      provider: map['provider'] ?? 'cpu',
      debug: map['debug'] ?? true,
    );
  }

  @override
  String toString() {
    return 'AudioTaggingModelConfig(zipformer: $zipformer, ced: $ced, numThreads: $numThreads, provider: $provider, debug: $debug)';
  }

  Map<String, dynamic> toJson() {
    return {
      'zipformer': zipformer.toJson(),
      'ced': ced,
      'numThreads': numThreads,
      'provider': provider,
      'debug': debug,
    };
  }

  final OfflineZipformerAudioTaggingModelConfig zipformer;
  final String ced;
  final int numThreads;
  final String provider;
  final bool debug;
}

class AudioTaggingConfig {
  AudioTaggingConfig({required this.model, this.labels = ''});

  factory AudioTaggingConfig.fromJson(Map<String, dynamic> map) {
    return AudioTaggingConfig(
      model: AudioTaggingModelConfig.fromJson(map['model']),
      labels: map['labels'] ?? '',
    );
  }

  @override
  String toString() {
    return 'AudioTaggingConfig(model: $model, labels: $labels)';
  }

  Map<String, dynamic> toJson() {
    return {
      'model': model.toJson(),
      'labels': labels,
    };
  }

  final AudioTaggingModelConfig model;
  final String labels;
}

class AudioEvent {
  AudioEvent({required this.name, required this.index, required this.prob});

  factory AudioEvent.fromJson(Map<String, dynamic> map) {
    return AudioEvent(
      name: map['name'],
      index: map['index'],
      prob: map['prob'],
    );
  }

  @override
  String toString() {
    return 'AudioEvent(name: $name, index: $index, prob: $prob)';
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'index': index,
      'prob': prob,
    };
  }

  final String name;
  final int index;
  final double prob;
}

class AudioTagging {
  AudioTagging.fromPtr({required this.ptr, required this.config});

  AudioTagging._({required this.ptr, required this.config});

  // The user has to invoke AudioTagging.free() to avoid memory leak.
  factory AudioTagging({required AudioTaggingConfig config}) {
    final c = calloc<SherpaOnnxAudioTaggingConfig>();

    final zipformerPtr = config.model.zipformer.model.toNativeUtf8();
    c.ref.model.zipformer.model = zipformerPtr;

    final cedPtr = config.model.ced.toNativeUtf8();
    c.ref.model.ced = cedPtr;

    c.ref.model.numThreads = config.model.numThreads;

    final providerPtr = config.model.provider.toNativeUtf8();
    c.ref.model.provider = providerPtr;

    c.ref.model.debug = config.model.debug ? 1 : 0;

    final labelsPtr = config.labels.toNativeUtf8();
    c.ref.labels = labelsPtr;

    if (SherpaOnnxBindings.sherpaOnnxCreateAudioTagging == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final ptr =
        SherpaOnnxBindings.sherpaOnnxCreateAudioTagging?.call(c) ?? nullptr;

    calloc.free(labelsPtr);
    calloc.free(providerPtr);
    calloc.free(cedPtr);
    calloc.free(zipformerPtr);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create audio tagging. Please check your config");
    }

    return AudioTagging._(ptr: ptr, config: config);
  }

  void free() {
    SherpaOnnxBindings.sherpaOnnxDestroyAudioTagging?.call(ptr);
    ptr = nullptr;
  }

  /// The user has to invoke stream.free() on the returned instance
  /// to avoid memory leak
  OfflineStream createStream() {
    final p = SherpaOnnxBindings.sherpaOnnxAudioTaggingCreateOfflineStream
            ?.call(ptr) ??
        nullptr;
    return OfflineStream(ptr: p);
  }

  List<AudioEvent> compute({required OfflineStream stream, required int topK}) {
    final pp = SherpaOnnxBindings.sherpaOnnxAudioTaggingCompute
            ?.call(ptr, stream.ptr, topK) ??
        nullptr;

    final ans = <AudioEvent>[];

    if (pp == nullptr) {
      return ans;
    }

    var i = 0;
    while (pp[i] != nullptr) {
      final p = pp[i];

      final name = p.ref.name.toDartString();
      final index = p.ref.index;
      final prob = p.ref.prob;
      final e = AudioEvent(name: name, index: index, prob: prob);
      ans.add(e);

      i += 1;
    }

    SherpaOnnxBindings.sherpaOnnxAudioTaggingFreeResults?.call(pp);

    return ans;
  }

  Pointer<SherpaOnnxAudioTagging> ptr;
  final AudioTaggingConfig config;
}
