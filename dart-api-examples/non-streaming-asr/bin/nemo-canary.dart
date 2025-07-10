// Copyright (c)  2025  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('encoder', help: 'Path to the NeMo Canary encoder model')
    ..addOption('decoder', help: 'Path to the NeMo Canary decoder model')
    ..addOption('src-lang', help: 'Language of the input audio')
    ..addOption('tgt-lang', help: 'Language of the recognition result')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['encoder'] == null ||
      res['decoder'] == null ||
      res['src-lang'] == null ||
      res['tgt-lang'] == null ||
      res['tokens'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final encoder = res['encoder'] as String;
  final decoder = res['decoder'] as String;
  final srcLang = res['src-lang'] as String;
  final tgtLang = res['tgt-lang'] as String;
  final tokens = res['tokens'] as String;
  final inputWav = res['input-wav'] as String;

  final canary = sherpa_onnx.OfflineCanaryModelConfig(
      encoder: encoder, decoder: decoder, srcLang: srcLang, tgtLang: tgtLang);

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    canary: canary,
    tokens: tokens,
    debug: false,
    numThreads: 1,
  );
  var config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final waveData = sherpa_onnx.readWave(inputWav);
  final stream = recognizer.createStream();

  stream.acceptWaveform(
      samples: waveData.samples, sampleRate: waveData.sampleRate);
  recognizer.decode(stream);

  final result = recognizer.getResult(stream);
  print('Result in $tgtLang: ${result.text}');

  stream.free();

  // Example to change the target language to de
  if (tgtLang != 'en') {
    var json = config.toJson();

    ((json['model'] as Map<String, dynamic>)!['canary']
        as Map<String, dynamic>)!['tgtLang'] = 'en';

    config = sherpa_onnx.OfflineRecognizerConfig.fromJson(json);
    recognizer.setConfig(config);

    final stream = recognizer.createStream();

    stream.acceptWaveform(
        samples: waveData.samples, sampleRate: waveData.sampleRate);
    recognizer.decode(stream);

    final result = recognizer.getResult(stream);
    print('Result in English: ${result.text}');
    stream.free();
  }

  recognizer.free();
}
