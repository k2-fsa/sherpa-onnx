// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';

import './offline_stream.dart';
import './sherpa_onnx_bindings.dart';

/// Offline audio tagging.
///
/// This module classifies complete audio clips and returns the most likely
/// events. See `dart-api-examples/audio-tagging/` for working examples.
///
/// Example:
///
/// ```dart
/// final modelConfig = AudioTaggingModelConfig(
///   zipformer: const OfflineZipformerAudioTaggingModelConfig(
///     model: './sherpa-onnx-zipformer-audio-tagging/model.int8.onnx',
///   ),
///   numThreads: 1,
///   debug: true,
/// );
///
/// final config = AudioTaggingConfig(
///   model: modelConfig,
///   labels: './sherpa-onnx-zipformer-audio-tagging/class_labels_indices.csv',
/// );
///
/// final tagger = AudioTagging(config: config);
/// final wave = readWave('./test.wav');
/// final stream = tagger.createStream();
/// stream.acceptWaveform(samples: wave.samples, sampleRate: wave.sampleRate);
/// final events = tagger.compute(stream: stream, topK: 5);
/// print(events);
/// stream.free();
/// tagger.free();
/// ```
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

/// Aggregate model configuration for audio tagging.
///
/// Configure either [zipformer] or [ced] for typical use.
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

/// Top-level configuration for [AudioTagging].
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

/// One predicted audio event.
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

/// Offline audio tagger.
class AudioTagging {
  AudioTagging.fromPtr({required this.ptr, required this.config});

  AudioTagging._({required this.ptr, required this.config});

  /// Create an audio tagger from [config].
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

  /// Release the native tagger.
  void free() {
    if (SherpaOnnxBindings.sherpaOnnxDestroyAudioTagging == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.sherpaOnnxDestroyAudioTagging?.call(ptr);
    ptr = nullptr;
  }

  /// Create an offline stream for one audio clip.
  OfflineStream createStream() {
    if (SherpaOnnxBindings.sherpaOnnxAudioTaggingCreateOfflineStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      throw Exception("Failed to create offline stream");
    }

    final p = SherpaOnnxBindings.sherpaOnnxAudioTaggingCreateOfflineStream
            ?.call(ptr) ??
        nullptr;

    if (p == nullptr) {
      throw Exception("Failed to create offline stream");
    }

    return OfflineStream(ptr: p);
  }

  /// Compute the top [topK] events for [stream].
  List<AudioEvent> compute({required OfflineStream stream, required int topK}) {
    if (SherpaOnnxBindings.sherpaOnnxAudioTaggingCompute == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return <AudioEvent>[];
    }

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
