// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('preprocessor',
        help: 'Path to the moonshine preprocessor model')
    ..addOption('encoder', help: 'Path to the moonshine encoder model')
    ..addOption('uncached-decoder',
        help: 'Path to moonshine uncached decoder model')
    ..addOption('cached-decoder',
        help: 'Path to moonshine cached decoder model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['preprocessor'] == null ||
      res['encoder'] == null ||
      res['uncached-decoder'] == null ||
      res['cached-decoder'] == null ||
      res['tokens'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final preprocessor = res['preprocessor'] as String;
  final encoder = res['encoder'] as String;
  final uncachedDecoder = res['uncached-decoder'] as String;
  final cachedDecoder = res['cached-decoder'] as String;
  final tokens = res['tokens'] as String;
  final inputWav = res['input-wav'] as String;

  final moonshine = sherpa_onnx.OfflineMoonshineModelConfig(
    preprocessor: preprocessor,
    encoder: encoder,
    uncachedDecoder: uncachedDecoder,
    cachedDecoder: cachedDecoder,
  );

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    moonshine: moonshine,
    tokens: tokens,
    debug: false,
    numThreads: 1,
  );
  final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final waveData = sherpa_onnx.readWave(inputWav);
  final stream = recognizer.createStream();

  stream.acceptWaveform(
      samples: waveData.samples, sampleRate: waveData.sampleRate);
  recognizer.decode(stream);

  final result = recognizer.getResult(stream);
  print(result.text);

  stream.free();
  recognizer.free();
}
