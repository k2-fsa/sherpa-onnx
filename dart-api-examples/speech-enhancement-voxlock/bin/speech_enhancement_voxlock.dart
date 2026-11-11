// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the speech enhancement API with VoxLock models.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock takes the loudest
// 3 seconds in the first 15 seconds of the input as the speaker to keep. See
// ../speech-enhancement-voxlock-with-voice-sample if you have a voice sample
// of the speaker.
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main(List<String> arguments) async {
  await sherpa_onnx.initBindingsAsync();

  final parser = ArgParser()
    ..addOption('enroll-model', help: 'Path to the VoxLock enroll model')
    ..addOption('chunk-model', help: 'Path to the VoxLock chunk model')
    ..addOption('input-wav', help: 'Path to input.wav')
    ..addOption('output-wav', help: 'Path to output.wav');

  final res = parser.parse(arguments);
  if (res['enroll-model'] == null ||
      res['chunk-model'] == null ||
      res['input-wav'] == null ||
      res['output-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final enrollModel = res['enroll-model'] as String;
  final chunkModel = res['chunk-model'] as String;
  final inputWav = res['input-wav'] as String;
  final outputWav = res['output-wav'] as String;

  final config = sherpa_onnx.OfflineSpeechDenoiserConfig(
      model: sherpa_onnx.OfflineSpeechDenoiserModelConfig(
    voxlock: sherpa_onnx.OfflineSpeechDenoiserVoxLockModelConfig(
      enrollModel: enrollModel,
      chunkModel: chunkModel,
    ),
    numThreads: 1,
    debug: true,
    provider: 'cpu',
  ));

  final sd = sherpa_onnx.OfflineSpeechDenoiser(config);

  final waveData = sherpa_onnx.readWave(inputWav);

  final begin = DateTime.now();
  final denoised =
      sd.run(samples: waveData.samples, sampleRate: waveData.sampleRate);
  final elapsedSeconds =
      DateTime.now().difference(begin).inMicroseconds / 1e6;

  sd.free();

  sherpa_onnx.writeWave(
      filename: outputWav,
      samples: denoised.samples,
      sampleRate: denoised.sampleRate);

  final duration = waveData.samples.length / waveData.sampleRate;
  print('Saved to $outputWav');
  print('Elapsed seconds: ${elapsedSeconds.toStringAsFixed(3)}');
  print('Audio duration in seconds: ${duration.toStringAsFixed(3)}');
  print('RTF: ${elapsedSeconds.toStringAsFixed(3)}/'
      '${duration.toStringAsFixed(3)} = '
      '${(elapsedSeconds / duration).toStringAsFixed(3)}');
}
