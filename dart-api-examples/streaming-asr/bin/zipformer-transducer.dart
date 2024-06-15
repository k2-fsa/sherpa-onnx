// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('encoder', help: 'Path to the encoder model')
    ..addOption('decoder', help: 'Path to decoder model')
    ..addOption('joiner', help: 'Path to joiner model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['encoder'] == null ||
      res['decoder'] == null ||
      res['joiner'] == null ||
      res['tokens'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final encoder = res['encoder'] as String;
  final decoder = res['decoder'] as String;
  final joiner = res['joiner'] as String;
  final tokens = res['tokens'] as String;
  final inputWav = res['input-wav'] as String;

  final transducer = sherpa_onnx.OnlineTransducerModelConfig(
    encoder: encoder,
    decoder: decoder,
    joiner: joiner,
  );

  final modelConfig = sherpa_onnx.OnlineModelConfig(
    transducer: transducer,
    tokens: tokens,
    debug: true,
    numThreads: 1,
  );
  final config = sherpa_onnx.OnlineRecognizerConfig(model: modelConfig);
  final recognizer = sherpa_onnx.OnlineRecognizer(config);

  final waveData = sherpa_onnx.readWave(inputWav);
  final stream = recognizer.createStream();

  // simulate streaming. You can choose an arbitrary chunk size.
  // chunkSize of a single sample is also ok, i.e, chunkSize = 1
  final chunkSize = 1600; // 0.1 second for 16kHz
  final numChunks = waveData.samples.length ~/ chunkSize;

  var last = '';
  for (int i = 0; i != numChunks; ++i) {
    int start = i * chunkSize;
    stream.acceptWaveform(
      samples:
          Float32List.sublistView(waveData.samples, start, start + chunkSize),
      sampleRate: waveData.sampleRate,
    );
    while (recognizer.isReady(stream)) {
      recognizer.decode(stream);
    }
    final result = recognizer.getResult(stream);
    if (result.text != last && result.text != '') {
      last = result.text;
      print(last);
    }
  }

  // 0.5 seconds, assume sampleRate is 16kHz
  final tailPaddings = Float32List(8000);
  stream.acceptWaveform(
    samples: tailPaddings,
    sampleRate: waveData.sampleRate,
  );

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }

  final result = recognizer.getResult(stream);

  if (result.text != '') {
    print(result.text);
  }

  stream.free();
  recognizer.free();
}
