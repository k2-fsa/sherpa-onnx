// Copyright (c)  2024  Xiaomi Corporation
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'dart:typed_data';
import "dart:io";

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './utils.dart';

Future<void> testNonStreamingParaformerAsr() async {
  var model = 'assets/sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx';
  var tokens = 'assets/sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt';
  var testWave = 'assets/sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav';

  model = await copyAssetFile(src: model, dst: 'model.int8.onnx');
  tokens = await copyAssetFile(src: tokens, dst: 'tokens.txt');
  testWave = await copyAssetFile(src: testWave, dst: '0.wav');

  final paraformer = sherpa_onnx.OfflineParaformerModelConfig(
    model: model,
  );

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    paraformer: paraformer,
    tokens: tokens,
    modelType: 'paraformer',
  );

  final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
  print(config);
  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final waveData = sherpa_onnx.readWave(testWave);
  final stream = recognizer.createStream();

  stream.acceptWaveform(
      samples: waveData.samples, sampleRate: waveData.sampleRate);
  recognizer.decode(stream);

  final result = recognizer.getResult(stream);
  print('result is: ${result}');

  print('recognizer: ${recognizer.ptr}');
  stream.free();
  recognizer.free();
}
