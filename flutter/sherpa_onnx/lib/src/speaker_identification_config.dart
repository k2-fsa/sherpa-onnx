// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for speaker identification -- no FFI, works on all platforms.

/// Speaker embedding extraction and speaker identification utilities.
///
/// See `dart-api-examples/speaker-identification/` for end-to-end examples.
///
/// Example:
///
/// ```dart
/// final extractor = SpeakerEmbeddingExtractor(
///   config: const SpeakerEmbeddingExtractorConfig(
///     model: './3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx',
///   ),
/// );
///
/// final stream = extractor.createStream();
/// stream.acceptWaveform(samples: wave.samples, sampleRate: wave.sampleRate);
/// while (extractor.isReady(stream)) {}
/// final embedding = extractor.compute(stream);
///
/// final manager = SpeakerEmbeddingManager(extractor.dim);
/// manager.add(name: 'alice', embedding: embedding);
/// print(manager.search(embedding: embedding, threshold: 0.6));
/// ```
class SpeakerEmbeddingExtractorConfig {
  const SpeakerEmbeddingExtractorConfig(
      {required this.model,
      this.numThreads = 1,
      this.debug = true,
      this.provider = 'cpu'});

  factory SpeakerEmbeddingExtractorConfig.fromJson(Map<String, dynamic> json) {
    return SpeakerEmbeddingExtractorConfig(
      model: json['model'] as String,
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'SpeakerEmbeddingExtractorConfig(model: $model, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final String model;
  final int numThreads;
  final bool debug;
  final String provider;
}
