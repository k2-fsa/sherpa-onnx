// Copyright (c)  2024  Xiaomi Corporation
// Shared config/data classes for offline speaker diarization -- no FFI, works on all platforms.

import './speaker_identification_config.dart';

/// Offline speaker diarization.
///
/// This module combines segmentation, speaker embedding extraction, and
/// clustering to assign speaker labels to time spans. See
/// `dart-api-examples/speaker-diarization/` for a complete example.
class OfflineSpeakerDiarizationSegment {
  const OfflineSpeakerDiarizationSegment({
    required this.start,
    required this.end,
    required this.speaker,
  });

  factory OfflineSpeakerDiarizationSegment.fromJson(Map<String, dynamic> json) {
    return OfflineSpeakerDiarizationSegment(
      start: (json['start'] as num).toDouble(),
      end: (json['end'] as num).toDouble(),
      speaker: json['speaker'] as int,
    );
  }

  @override
  String toString() {
    return 'OfflineSpeakerDiarizationSegment(start: $start, end: $end, speaker: $speaker)';
  }

  Map<String, dynamic> toJson() => {
        'start': start,
        'end': end,
        'speaker': speaker,
      };

  final double start;
  final double end;
  final int speaker;
}

/// Pyannote segmentation model path.
class OfflineSpeakerSegmentationPyannoteModelConfig {
  const OfflineSpeakerSegmentationPyannoteModelConfig({
    this.model = '',
    this.windowShiftRatio = 0.1,
  });

  factory OfflineSpeakerSegmentationPyannoteModelConfig.fromJson(
      Map<String, dynamic> json) {
    return OfflineSpeakerSegmentationPyannoteModelConfig(
      model: json['model'] as String? ?? '',
      windowShiftRatio:
          (json['windowShiftRatio'] as num?)?.toDouble() ?? 0.1,
    );
  }

  @override
  String toString() {
    return 'OfflineSpeakerSegmentationPyannoteModelConfig(model: $model, '
        'windowShiftRatio: $windowShiftRatio)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'windowShiftRatio': windowShiftRatio,
      };

  final String model;
  final double windowShiftRatio;
}

/// Segmentation model configuration for speaker diarization.
class OfflineSpeakerSegmentationModelConfig {
  const OfflineSpeakerSegmentationModelConfig({
    this.pyannote = const OfflineSpeakerSegmentationPyannoteModelConfig(),
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
  });

  factory OfflineSpeakerSegmentationModelConfig.fromJson(
      Map<String, dynamic> json) {
    return OfflineSpeakerSegmentationModelConfig(
      pyannote: json['pyannote'] != null
          ? OfflineSpeakerSegmentationPyannoteModelConfig.fromJson(
              json['pyannote'] as Map<String, dynamic>)
          : const OfflineSpeakerSegmentationPyannoteModelConfig(),
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'OfflineSpeakerSegmentationModelConfig(pyannote: $pyannote, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'pyannote': pyannote.toJson(),
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final OfflineSpeakerSegmentationPyannoteModelConfig pyannote;

  final int numThreads;
  final bool debug;
  final String provider;
}

/// Clustering options used after segmentation and embedding extraction.
class FastClusteringConfig {
  const FastClusteringConfig({
    this.numClusters = -1,
    this.threshold = 0.5,
  });

  factory FastClusteringConfig.fromJson(Map<String, dynamic> json) {
    return FastClusteringConfig(
      numClusters: json['numClusters'] as int? ?? -1,
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
    );
  }

  @override
  String toString() {
    return 'FastClusteringConfig(numClusters: $numClusters, threshold: $threshold)';
  }

  Map<String, dynamic> toJson() => {
        'numClusters': numClusters,
        'threshold': threshold,
      };

  final int numClusters;
  final double threshold;
}

/// Top-level configuration for [OfflineSpeakerDiarization].
class OfflineSpeakerDiarizationConfig {
  const OfflineSpeakerDiarizationConfig({
    this.segmentation = const OfflineSpeakerSegmentationModelConfig(),
    this.embedding = const SpeakerEmbeddingExtractorConfig(model: ''),
    this.clustering = const FastClusteringConfig(),
    this.minDurationOn = 0.2,
    this.minDurationOff = 0.5,
  });

  factory OfflineSpeakerDiarizationConfig.fromJson(Map<String, dynamic> json) {
    return OfflineSpeakerDiarizationConfig(
      segmentation: json['segmentation'] != null
          ? OfflineSpeakerSegmentationModelConfig.fromJson(
              json['segmentation'] as Map<String, dynamic>)
          : const OfflineSpeakerSegmentationModelConfig(),
      embedding: json['embedding'] != null
          ? SpeakerEmbeddingExtractorConfig.fromJson(
              json['embedding'] as Map<String, dynamic>)
          : const SpeakerEmbeddingExtractorConfig(model: ''),
      clustering: json['clustering'] != null
          ? FastClusteringConfig.fromJson(
              json['clustering'] as Map<String, dynamic>)
          : const FastClusteringConfig(),
      minDurationOn: (json['minDurationOn'] as num?)?.toDouble() ?? 0.2,
      minDurationOff: (json['minDurationOff'] as num?)?.toDouble() ?? 0.5,
    );
  }

  @override
  String toString() {
    return 'OfflineSpeakerDiarizationConfig(segmentation: $segmentation, embedding: $embedding, clustering: $clustering, minDurationOn: $minDurationOn, minDurationOff: $minDurationOff)';
  }

  Map<String, dynamic> toJson() => {
        'segmentation': segmentation.toJson(),
        'embedding': embedding.toJson(),
        'clustering': clustering.toJson(),
        'minDurationOn': minDurationOn,
        'minDurationOff': minDurationOff,
      };

  final OfflineSpeakerSegmentationModelConfig segmentation;
  final SpeakerEmbeddingExtractorConfig embedding;
  final FastClusteringConfig clustering;
  final double minDurationOff; // in seconds
  final double minDurationOn; // in seconds
}
