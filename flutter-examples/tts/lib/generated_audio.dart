// Copyright (c)  2026  Xiaomi Corporation
import 'dart:typed_data';

/// Represents a generated audio item with metadata.
class GeneratedAudioItem {
  /// Short label derived from the input text.
  final String label;

  /// WAV file bytes (non-null on web, null on native where file is on disk).
  final Uint8List? wavBytes;

  /// File path on disk (non-null on native, null on web).
  final String? filePath;

  /// Duration of the generated audio in seconds.
  final double duration;

  /// Time taken to generate the audio in seconds.
  final double elapsed;

  /// Sample rate of the audio.
  final int sampleRate;

  /// Generation ID to distinguish from previous generations.
  final int generationId;

  GeneratedAudioItem({
    required this.label,
    this.wavBytes,
    this.filePath,
    required this.duration,
    required this.elapsed,
    required this.sampleRate,
    this.generationId = 0,
  });

  /// Create a label from input text (first 30 characters).
  static String makeLabel(String text) {
    final trimmed = text.trim();
    if (trimmed.length <= 30) return trimmed;
    return '${trimmed.substring(0, 27)}...';
  }
}

/// A chunk of audio samples received during streaming generation.
class AudioChunk {
  /// PCM audio samples (Float32, mono).
  final Float32List samples;

  /// Progress of generation (0.0 to 1.0).
  final double progress;

  /// Sample rate of the audio.
  final int sampleRate;

  /// Generation ID to distinguish from previous generations.
  final int generationId;

  AudioChunk({
    required this.samples,
    required this.progress,
    required this.sampleRate,
    this.generationId = 0,
  });
}
