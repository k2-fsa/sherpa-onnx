// Copyright (c)  2026  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('model', help: 'Path to model.onnx')
    ..addOption('bpe-vocab', help: 'Path to bpe.vocab');

  final res = parser.parse(arguments);
  if (res['model'] == null || res['bpe-vocab'] == null) {
    print(parser.usage);
    exit(1);
  }

  final modelFile = res['model'] as String;
  final bpeVocab = res['bpe-vocab'] as String;
  final modelConfig = sherpa_onnx.OnlinePunctuationModelConfig(
    cnnBiLstm: modelFile,
    bpeVocab: bpeVocab,
    numThreads: 1,
    provider: 'cpu',
    debug: false,
  );

  final config = sherpa_onnx.OnlinePunctuationConfig(model: modelConfig);
  final punct = sherpa_onnx.OnlinePunctuation(config: config);

  final texts = [
    'how are you i am fine thank you',
    'The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry',
  ];

  for (final t in texts) {
    final textWithPunct = punct.addPunct(t);
    print('----------');
    print('Before: $t');
    print('After: $textWithPunct');
  }
  print('----------');

  punct.free();
}
