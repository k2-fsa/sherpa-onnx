// Copyright (c)  2024  Xiaomi Corporation
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'dart:typed_data';
import "dart:io";

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './utils.dart';

Future<void> testStreamingTransducerAsr() async {
  var encoder =
      'assets/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx';
  var decoder =
      'assets/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx';
  var joiner =
      'assets/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx';
  var tokens =
      'assets/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt';

  var testWave =
      'assets/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav';

  encoder = await copyAssetFile(src: encoder, dst: 'encoder.onnx');
  decoder = await copyAssetFile(src: decoder, dst: 'decoder.onnx');
  joiner = await copyAssetFile(src: joiner, dst: 'joiner.onnx');
  tokens = await copyAssetFile(src: tokens, dst: 'tokens.txt');
  testWave = await copyAssetFile(src: testWave, dst: 'test.wav');

  final transducer = sherpa_onnx.OnlineTransducerModelConfig(
    encoder: encoder,
    decoder: decoder,
    joiner: joiner,
  );

  final modelConfig = sherpa_onnx.OnlineModelConfig(
    transducer: transducer,
    tokens: tokens,
    modelType: 'zipformer',
  );

  final config = sherpa_onnx.OnlineRecognizerConfig(model: modelConfig);
  print(config);
  final recognizer = sherpa_onnx.OnlineRecognizer(config);

  final waveData = sherpa_onnx.readWave(testWave);
  final stream = recognizer.createStream();

  stream.acceptWaveform(
      samples: waveData.samples, sampleRate: waveData.sampleRate);
  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }

  final result = recognizer.getResult(stream);
  print('result is: ${result}');

  print('recognizer: ${recognizer.ptr}');
  stream.free();
  recognizer.free();
}
