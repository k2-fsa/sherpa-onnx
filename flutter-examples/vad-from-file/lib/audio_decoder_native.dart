// Copyright (c)  2026  Xiaomi Corporation
// Native audio decoder — uses FFmpeg to decode any audio/video format to 16kHz mono PCM.
import 'dart:io';
import 'dart:typed_data';

import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';
import 'package:path_provider/path_provider.dart';

import './resample.dart';

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

/// Decode audio bytes to 16kHz mono Float32 PCM samples using FFmpeg.
/// Returns null if decoding fails.
Future<DecodedAudio?> decodeAudioBytes(Uint8List bytes) async {
  try {
    final tempDir = await getTemporaryDirectory();
    await tempDir.create(recursive: true);
    final inputPath = '${tempDir.path}/vad_input_${DateTime.now().microsecondsSinceEpoch}';
    final file = File(inputPath);
    await file.writeAsBytes(bytes);
    final result = await _decodePath(inputPath);
    try { await file.delete(); } catch (_) {}
    return result;
  } catch (e) {
    print('Audio decode error: $e');
    return null;
  }
}

/// Decode an audio file to 16kHz mono Float32 PCM samples using FFmpeg.
/// Returns null if decoding fails.
Future<DecodedAudio?> decodeAudioFile(String filePath) async {
  return _decodePath(filePath);
}

Future<DecodedAudio?> _decodePath(String filePath) async {
  try {
    final tempDir = await getTemporaryDirectory();
    await tempDir.create(recursive: true);
    final outputPath =
        '${tempDir.path}/decoded_${DateTime.now().microsecondsSinceEpoch}.raw';

    // Use FFmpeg to convert any audio/video to 16kHz mono Float32 PCM.
    final command =
        '-i "$filePath" -ar 16000 -ac 1 -f f32le -acodec pcm_f32le -y "$outputPath"';

    final session = await FFmpegKit.execute(command);
    final returnCode = await session.getReturnCode();

    if (!ReturnCode.isSuccess(returnCode)) {
      final logs = await session.getOutput();
      print('FFmpeg error: $logs');
      return null;
    }

    final outFile = File(outputPath);
    if (!await outFile.exists()) {
      print('FFmpeg output file not found: $outputPath');
      return null;
    }

    final bytes = await outFile.readAsBytes();
    await outFile.delete();

    if (bytes.length < 4) return null;

    final numSamples = bytes.length ~/ 4;
    final samples = Float32List(numSamples);
    final bd = bytes.buffer.asByteData(bytes.offsetInBytes, bytes.lengthInBytes);
    for (int i = 0; i < numSamples; i++) {
      samples[i] = bd.getFloat32(i * 4, Endian.little);
    }

    final duration = numSamples / 16000.0;

    return DecodedAudio(
      samples: samples,
      sampleRate: 16000,
      duration: duration,
    );
  } catch (e) {
    print('Audio decode error: $e');
    return null;
  }
}
