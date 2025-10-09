// Copyright (c)  2025  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  print('sherpa-onnx version: ${sherpa_onnx.getVersion()}');
  print('sherpa-onnx gitSha1: ${sherpa_onnx.getGitSha1()}');
  print('sherpa-onnx gitDate: ${sherpa_onnx.getGitDate()}');

  final parser = ArgParser()
    ..addOption('model', help: 'Path to the SenseVoice model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('language',
        help: 'auto, zh, en, ja, ko, yue, or leave it empty to use auto',
        defaultsTo: '')
    ..addOption('use-itn',
        help: 'true to use inverse text normalization', defaultsTo: 'false')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe')
    ..addOption('hr-lexicon',
        help: 'Path to lexicon.txt for homophone replacer')
    ..addOption('hr-rule-fsts',
        help: 'Path to replace.fst for homophone replacer');

  final res = parser.parse(arguments);
  if (res['model'] == null ||
      res['tokens'] == null ||
      res['hr-lexicon'] == null ||
      res['hr-rule-fsts'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final model = res['model'] as String;
  final tokens = res['tokens'] as String;
  final inputWav = res['input-wav'] as String;
  final language = res['language'] as String;
  final useItn = (res['use-itn'] as String).toLowerCase() == 'true';
  final hrLexicon = res['hr-lexicon'] as String;
  final hrRuleFsts = res['hr-rule-fsts'] as String;

  final senseVoice = sherpa_onnx.OfflineSenseVoiceModelConfig(
      model: model, language: language, useInverseTextNormalization: useItn);

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    senseVoice: senseVoice,
    tokens: tokens,
    debug: true,
    numThreads: 1,
  );

  final hr = sherpa_onnx.HomophoneReplacerConfig(
      lexicon: hrLexicon, ruleFsts: hrRuleFsts);

  final config =
      sherpa_onnx.OfflineRecognizerConfig(model: modelConfig, hr: hr);

  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final waveData = sherpa_onnx.readWave(inputWav);
  final stream = recognizer.createStream();

  stream.acceptWaveform(
      samples: waveData.samples, sampleRate: waveData.sampleRate);
  recognizer.decode(stream);

  final result = recognizer.getResult(stream);
  print(result.text);

  stream.free();
  recognizer.free();
}
