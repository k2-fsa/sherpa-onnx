// Copyright (c)  2025  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('model', help: 'Path to gtcrn onnx model')
    ..addOption('input-wav', help: 'Path to input.wav')
    ..addOption('output-wav', help: 'Path to output.wav');

  final res = parser.parse(arguments);
  if (res['model'] == null ||
      res['input-wav'] == null ||
      res['output-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final model = res['model'] as String;
  final inputWav = res['input-wav'] as String;
  final outputWav = res['output-wav'] as String;

  final config = sherpa_onnx.OfflineSpeechDenoiserConfig(
      model: sherpa_onnx.OfflineSpeechDenoiserModelConfig(
    gtcrn: sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model: model),
    numThreads: 1,
    debug: true,
    provider: 'cpu',
  ));

  final sd = sherpa_onnx.OfflineSpeechDenoiser(config);

  final waveData = sherpa_onnx.readWave(inputWav);

  final denoised =
      sd.run(samples: waveData.samples, sampleRate: waveData.sampleRate);

  sd.free();

  sherpa_onnx.writeWave(
      filename: outputWav,
      samples: denoised.samples,
      sampleRate: denoised.sampleRate);

  print('Saved to $outputWav');
}
