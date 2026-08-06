// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for audio tagging -- no FFI, works on all platforms.

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
