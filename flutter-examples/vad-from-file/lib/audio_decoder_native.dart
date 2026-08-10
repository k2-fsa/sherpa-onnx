// Copyright (c)  2026  Xiaomi Corporation
// Native audio decoder — uses sherpa_onnx readWave for WAV files.
import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

/// Result of decoding an audio file.
class DecodedAudio {
  final Float32List samples;
  final int sampleRate;
  final double duration;
  const DecodedAudio({
    required this.samples,
    required this.sampleRate,
    required this.duration,
  });
}

/// Decode audio bytes to Float32 PCM samples.
/// On native, supports WAV files via sherpa_onnx readWave.
/// Returns null if decoding fails.
Future<DecodedAudio?> decodeAudioBytes(Uint8List bytes) async {
  try {
    sherpa_onnx.initBindings();

    final tempDir = await getTemporaryDirectory();
    await tempDir.create(recursive: true);
    final inputPath = '${tempDir.path}/vad_input_${DateTime.now().microsecondsSinceEpoch}.wav';
    final file = File(inputPath);
    await file.writeAsBytes(bytes);

    final waveData = sherpa_onnx.readWave(inputPath);
    try { await file.delete(); } catch (_) {}

    if (waveData.samples.isEmpty) return null;

    return DecodedAudio(
      samples: waveData.samples,
      sampleRate: waveData.sampleRate,
      duration: waveData.samples.length / waveData.sampleRate,
    );
  } catch (e) {
    print('Audio decode error: $e');
    return null;
  }
}

/// Decode an audio file to Float32 PCM samples.
/// Returns null if decoding fails.
Future<DecodedAudio?> decodeAudioFile(String filePath) async {
  try {
    sherpa_onnx.initBindings();

    final waveData = sherpa_onnx.readWave(filePath);
    if (waveData.samples.isEmpty) return null;

    return DecodedAudio(
      samples: waveData.samples,
      sampleRate: waveData.sampleRate,
      duration: waveData.samples.length / waveData.sampleRate,
    );
  } catch (e) {
    print('Audio decode error: $e');
    return null;
  }
}
