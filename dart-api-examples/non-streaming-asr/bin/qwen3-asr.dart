// Copyright (c)  2026  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('conv-frontend', help: 'Path to the conv frontend model')
    ..addOption('encoder', help: 'Path to the encoder model')
    ..addOption('decoder', help: 'Path to the decoder model')
    ..addOption('tokenizer', help: 'Path to the tokenizer directory')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['conv-frontend'] == null ||
      res['encoder'] == null ||
      res['decoder'] == null ||
      res['tokenizer'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final convFrontend = res['conv-frontend'] as String;
  final encoder = res['encoder'] as String;
  final decoder = res['decoder'] as String;
  final tokenizer = res['tokenizer'] as String;
  final inputWav = res['input-wav'] as String;

  final qwen3Asr = sherpa_onnx.OfflineQwen3AsrModelConfig(
    convFrontend: convFrontend,
    encoder: encoder,
    decoder: decoder,
    tokenizer: tokenizer,
  );

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    qwen3Asr: qwen3Asr,
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

  final start = DateTime.now().millisecondsSinceEpoch;
  recognizer.decode(stream);
  final stop = DateTime.now().millisecondsSinceEpoch;

  final result = recognizer.getResult(stream);
  print(result.text);

  final timeElapsedSeconds = (stop - start) / 1000.0;
  final audioDuration = waveData.samples.length / waveData.sampleRate;
  final realTimeFactor = timeElapsedSeconds / audioDuration;

  print('-- elapsed : ${timeElapsedSeconds.toStringAsFixed(3)} seconds');
  print('-- audio duration: ${audioDuration.toStringAsFixed(3)} seconds');
  print('-- real-time factor (RTF): ${realTimeFactor.toStringAsFixed(3)}');

  stream.free();
  recognizer.free();
}
