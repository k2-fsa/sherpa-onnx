// Copyright (c)  2026  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('encoder', help: 'Path to the encoder model')
    ..addOption('decoder', help: 'Path to the decoder model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe')
    ..addOption('language', help: 'Language to decode', defaultsTo: 'en');

  final res = parser.parse(arguments);
  if (res['encoder'] == null ||
      res['decoder'] == null ||
      res['tokens'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final encoder = res['encoder'] as String;
  final decoder = res['decoder'] as String;
  final tokens = res['tokens'] as String;
  final inputWav = res['input-wav'] as String;
  final language = res['language'] as String;

  final cohereTranscribe = sherpa_onnx.OfflineCohereTranscribeModelConfig(
    encoder: encoder,
    decoder: decoder,
    usePunct: true,
    useItn: true,
  );

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    cohereTranscribe: cohereTranscribe,
    tokens: tokens,
    debug: true,
    numThreads: 1,
  );
  final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final waveData = sherpa_onnx.readWave(inputWav);
  final stream = recognizer.createStream();
  stream.setOption(key: 'language', value: language);
  stream.acceptWaveform(
    samples: waveData.samples,
    sampleRate: waveData.sampleRate,
  );

  recognizer.decode(stream);

  final result = recognizer.getResult(stream);
  print(result.text);

  stream.free();
  recognizer.free();
}
