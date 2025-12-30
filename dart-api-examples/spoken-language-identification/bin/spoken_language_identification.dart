// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('encoder', help: 'Path to the whisper encoder model')
    ..addOption('decoder', help: 'Path to the whisper decoder model')
    ..addOption('tail-paddings', help: 'Tail paddings for the whisper model', defaultsTo: '0')
    ..addOption('wav', help: 'Path to test.wav for language identification')
    ..addFlag('help', abbr: 'h', help: 'Show this help message', negatable: false);

  final res = parser.parse(arguments);
  if (res['help'] as bool) {
    print(parser.usage);
    exit(0);
  }

  if (res['encoder'] == null || res['decoder'] == null || res['wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final encoder = res['encoder'] as String;
  final decoder = res['decoder'] as String;
  final tailPaddings = int.tryParse(res['tail-paddings'] as String) ?? 0;
  final wav = res['wav'] as String;

  final whisperConfig = sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
    encoder: encoder,
    decoder: decoder,
    tailPaddings: tailPaddings,
  );

  final config = sherpa_onnx.SpokenLanguageIdentificationConfig(
    whisper: whisperConfig,
    numThreads: 1,
    debug: true,
    provider: 'cpu',
  );

  final slid = sherpa_onnx.SpokenLanguageIdentification(config);

  final waveData = sherpa_onnx.readWave(wav);

  final stream = slid.createStream();
  stream.acceptWaveform(samples: waveData.samples, sampleRate: waveData.sampleRate);

  final result = slid.compute(stream);

  print('File: $wav');
  print('Detected language: ${result.lang}');

  stream.free();
  slid.free();
}
