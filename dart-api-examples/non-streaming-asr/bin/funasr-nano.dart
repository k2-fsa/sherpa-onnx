// Copyright (c)  2026  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('encoder-adaptor', help: 'Path to the encoder adaptor model')
    ..addOption('llm', help: 'Path to the llm model')
    ..addOption('embedding', help: 'Path to the embedding model')
    ..addOption('tokenizer', help: 'Path to the tokenizer directory')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['encoder-adaptor'] == null ||
      res['llm'] == null ||
      res['embedding'] == null ||
      res['tokenizer'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final encoderAdaptor = res['encoder-adaptor'] as String;
  final llm = res['llm'] as String;
  final embedding = res['embedding'] as String;
  final tokenizer = res['tokenizer'] as String;
  final inputWav = res['input-wav'] as String;

  final funasrNano = sherpa_onnx.OfflineFunAsrNanoModelConfig(
    encoderAdaptor: encoderAdaptor,
    llm: llm,
    embedding: embedding,
    tokenizer: tokenizer,
  );

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    funasrNano: funasrNano,
    tokens: '',
    debug: true,
    numThreads: 1,
  );
  final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final waveData = sherpa_onnx.readWave(inputWav);
  final stream = recognizer.createStream();

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
