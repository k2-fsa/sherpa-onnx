// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a non-streaming Zipformer CTC model with an HLG
// decoding graph via sherpa-onnx's Dart API.
//
// The HLG graph is passed to the C API through ctcFstDecoderConfig.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
//
// dart run bin/zipformer-ctc-hlg.dart \
//   --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx \
//   --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
//   --graph ./sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst \
//   --input-wav ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav
//
// clang-format on
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main(List<String> arguments) async {
  await sherpa_onnx.initBindingsAsync();

  final parser = ArgParser()
    ..addOption('model', help: 'Path to the Zipformer CTC model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('graph', help: 'Path to H.fst, HL.fst or HLG.fst')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['model'] == null ||
      res['tokens'] == null ||
      res['graph'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final model = res['model'] as String;
  final tokens = res['tokens'] as String;
  final graph = res['graph'] as String;
  final inputWav = res['input-wav'] as String;

  final zipformerCtc = sherpa_onnx.OfflineZipformerCtcModelConfig(model: model);

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    zipformerCtc: zipformerCtc,
    tokens: tokens,
    debug: true,
    numThreads: 1,
  );

  // This is what the example exercises: the graph and maxActive reach the C
  // API through ctcFstDecoderConfig.
  final config = sherpa_onnx.OfflineRecognizerConfig(
    model: modelConfig,
    ctcFstDecoderConfig: sherpa_onnx.OfflineCtcFstDecoderConfig(
      graph: graph,
      maxActive: 3000,
    ),
  );
  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final waveData = sherpa_onnx.readWave(inputWav);
  final stream = recognizer.createStream();

  stream.acceptWaveform(
      samples: waveData.samples, sampleRate: waveData.sampleRate);
  recognizer.decode(stream);

  final result = recognizer.getResult(stream);
  print(result.text);
  print(result);

  stream.free();
  recognizer.free();
}
