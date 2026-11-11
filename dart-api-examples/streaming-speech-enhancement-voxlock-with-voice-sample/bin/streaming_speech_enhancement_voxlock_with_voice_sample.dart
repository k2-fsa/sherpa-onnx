// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the online speech enhancement API with VoxLock
// models and a voice sample.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example provides a voice sample of the speaker to keep, so VoxLock
// isolates that speaker from the first sample. The voice sample must have at
// least 3 seconds of the speaker's voice; from a longer file, VoxLock uses the
// loudest 3 seconds within its first 30 seconds. See
// ../streaming-speech-enhancement-voxlock if you don't have a voice sample.
import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main(List<String> arguments) async {
  await sherpa_onnx.initBindingsAsync();

  final parser = ArgParser()
    ..addOption('enroll-model', help: 'Path to the VoxLock enroll model')
    ..addOption('chunk-model', help: 'Path to the VoxLock chunk model')
    ..addOption('voice-sample', help: 'Path to a wave file with the voice of the speaker to keep')
    ..addOption('input-wav', help: 'Path to input.wav')
    ..addOption('output-wav', help: 'Path to output.wav');

  final res = parser.parse(arguments);
  if (res['enroll-model'] == null ||
      res['chunk-model'] == null ||
      res['voice-sample'] == null ||
      res['input-wav'] == null ||
      res['output-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final enrollModel = res['enroll-model'] as String;
  final chunkModel = res['chunk-model'] as String;
  final voiceSample = res['voice-sample'] as String;
  final inputWav = res['input-wav'] as String;
  final outputWav = res['output-wav'] as String;

  final config = sherpa_onnx.OnlineSpeechDenoiserConfig(
      model: sherpa_onnx.OfflineSpeechDenoiserModelConfig(
    voxlock: sherpa_onnx.OfflineSpeechDenoiserVoxLockModelConfig(
      enrollModel: enrollModel,
      chunkModel: chunkModel,
      voiceSample: voiceSample,
    ),
    numThreads: 1,
    debug: true,
    provider: 'cpu',
  ));

  final sd = sherpa_onnx.OnlineSpeechDenoiser(config);
  final waveData = sherpa_onnx.readWave(inputWav);
  final frameShift = sd.frameShiftInSamples;
  final output = <double>[];

  final begin = DateTime.now();
  var start = 0;
  while (start < waveData.samples.length) {
    final end = start + frameShift < waveData.samples.length
        ? start + frameShift
        : waveData.samples.length;
    final chunk = waveData.samples.sublist(start, end);
    final denoised = sd.run(samples: chunk, sampleRate: waveData.sampleRate);
    output.addAll(denoised.samples);
    start = end;
  }

  output.addAll(sd.flush().samples);
  final elapsedSeconds = DateTime.now().difference(begin).inMicroseconds / 1e6;
  sd.free();

  sherpa_onnx.writeWave(
    filename: outputWav,
    samples: Float32List.fromList(output),
    sampleRate: waveData.sampleRate,
  );

  final duration = waveData.samples.length / waveData.sampleRate;
  print('Saved to $outputWav');
  print('Elapsed seconds: ${elapsedSeconds.toStringAsFixed(3)}');
  print('Audio duration in seconds: ${duration.toStringAsFixed(3)}');
  print('RTF: ${elapsedSeconds.toStringAsFixed(3)}/'
      '${duration.toStringAsFixed(3)} = '
      '${(elapsedSeconds / duration).toStringAsFixed(3)}');
}
