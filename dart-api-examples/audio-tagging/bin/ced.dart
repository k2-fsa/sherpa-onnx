// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('model', help: 'Path to the zipformer model')
    ..addOption('labels', help: 'Path to class_labels_indices.csv')
    ..addOption('top-k', help: 'topK events to be returned', defaultsTo: '5')
    ..addOption('wav', help: 'Path to test.wav to be tagged');

  final res = parser.parse(arguments);
  if (res['model'] == null || res['labels'] == null || res['wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final model = res['model'] as String;
  final labels = res['labels'] as String;
  final topK = int.tryParse(res['top-k'] as String) ?? 5;
  final wav = res['wav'] as String;

  final modelConfig = sherpa_onnx.AudioTaggingModelConfig(
    ced: model,
    numThreads: 1,
    debug: true,
    provider: 'cpu',
  );

  final config = sherpa_onnx.AudioTaggingConfig(
    model: modelConfig,
    labels: labels,
  );

  final at = sherpa_onnx.AudioTagging(config: config);

  final waveData = sherpa_onnx.readWave(wav);

  final stream = at.createStream();
  stream.acceptWaveform(
      samples: waveData.samples, sampleRate: waveData.sampleRate);

  final events = at.compute(stream: stream, topK: topK);

  print(events);

  stream.free();
  at.free();
}
