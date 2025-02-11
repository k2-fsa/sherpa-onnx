// Copyright (c)  2025  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('model', help: 'Path to the onnx model')
    ..addOption('voices', help: 'Path to the voices.bin')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption(
      'data-dir',
      help: 'Path to espeak-ng-data directory',
      defaultsTo: '',
    )
    ..addOption('rule-fsts', help: 'Path to rule fsts', defaultsTo: '')
    ..addOption('rule-fars', help: 'Path to rule fars', defaultsTo: '')
    ..addOption('text', help: 'Text to generate TTS for')
    ..addOption('output-wav', help: 'Filename to save the generated audio')
    ..addOption('speed', help: 'Speech speed', defaultsTo: '1.0')
    ..addOption(
      'sid',
      help: 'Speaker ID to select. Used only for multi-speaker TTS',
      defaultsTo: '0',
    );
  final res = parser.parse(arguments);
  if (res['model'] == null ||
      res['voices'] == null ||
      res['tokens'] == null ||
      res['data-dir'] == null ||
      res['output-wav'] == null ||
      res['text'] == null) {
    print(parser.usage);
    exit(1);
  }
  final model = res['model'] as String;
  final voices = res['voices'] as String;
  final tokens = res['tokens'] as String;
  final dataDir = res['data-dir'] as String;
  final ruleFsts = res['rule-fsts'] as String;
  final ruleFars = res['rule-fars'] as String;
  final text = res['text'] as String;
  final outputWav = res['output-wav'] as String;
  var speed = double.tryParse(res['speed'] as String) ?? 1.0;
  final sid = int.tryParse(res['sid'] as String) ?? 0;

  if (speed == 0) {
    speed = 1.0;
  }

  final kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
    model: model,
    voices: voices,
    tokens: tokens,
    dataDir: dataDir,
    lengthScale: 1 / speed,
  );

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    kokoro: kokoro,
    numThreads: 1,
    debug: true,
  );
  final config = sherpa_onnx.OfflineTtsConfig(
    model: modelConfig,
    maxNumSenetences: 1,
    ruleFsts: ruleFsts,
    ruleFars: ruleFars,
  );

  final tts = sherpa_onnx.OfflineTts(config);
  final audio = tts.generate(text: text, sid: sid, speed: speed);
  tts.free();

  sherpa_onnx.writeWave(
    filename: outputWav,
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  );
  print('Saved to $outputWav');
}
