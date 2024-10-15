// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('model', help: 'Path to the SenseVoice model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('language',
        help: 'auto, zh, en, ja, ko, yue, or leave it empty to use auto',
        defaultsTo: '')
    ..addOption('use-itn',
        help: 'true to use inverse text normalization', defaultsTo: 'false')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['model'] == null ||
      res['tokens'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final model = res['model'] as String;
  final tokens = res['tokens'] as String;
  final inputWav = res['input-wav'] as String;
  final language = res['language'] as String;
  final useItn = (res['use-itn'] as String).toLowerCase() == 'true';

  final senseVoice = sherpa_onnx.OfflineSenseVoiceModelConfig(
      model: model, language: language, useInverseTextNormalization: useItn);

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    senseVoice: senseVoice,
    tokens: tokens,
    debug: true,
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
