// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()..addOption('model', help: 'Path to model.onnx');

  final res = parser.parse(arguments);
  if (res['model'] == null) {
    print(parser.usage);
    exit(1);
  }

  final modelFile = res['model'] as String;
  final modelConfig = sherpa_onnx.OfflinePunctuationModelConfig(
    ctTransformer: modelFile,
    numThreads: 1,
    provider: 'cpu',
    debug: false,
  );

  final config = sherpa_onnx.OfflinePunctuationConfig(model: modelConfig);

  final punct = sherpa_onnx.OfflinePunctuation(config: config);

  final texts = [
    '这是一个测试你好吗How are you我很好thank you are you ok谢谢你',
    '我们都是木头人不会说话不会动',
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
