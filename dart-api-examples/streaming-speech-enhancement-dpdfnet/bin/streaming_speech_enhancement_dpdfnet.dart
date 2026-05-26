// Copyright (c)  2026  Xiaomi Corporation
import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('model', help: 'Path to a DPDFNet onnx model')
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

  final config = sherpa_onnx.OnlineSpeechDenoiserConfig(
    model: sherpa_onnx.OfflineSpeechDenoiserModelConfig(
      gtcrn: const sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(),
      dpdfnet:
          sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig(model: model),
      numThreads: 1,
      debug: true,
      provider: 'cpu',
    ),
  );

  final sd = sherpa_onnx.OnlineSpeechDenoiser(config);
  final waveData = sherpa_onnx.readWave(inputWav);
  final frameShift = sd.frameShiftInSamples;
  final output = <double>[];

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
  sd.free();

  sherpa_onnx.writeWave(
    filename: outputWav,
    samples: Float32List.fromList(output),
    sampleRate: waveData.sampleRate,
  );

  print('Saved to $outputWav');
}
