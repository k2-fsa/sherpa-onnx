// Copyright (c)  2025  Xiaomi Corporation
//
// Phonemize approach: the text is converted to phoneme code points with the
// piper_phonemize package and passed to sherpa-onnx. No lexicon file and no
// espeak-ng-data for sherpa-onnx are needed. See ./matcha-en-lexicon.dart for
// the lexicon-based twin.
import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:piper_phonemize/piper_phonemize.dart' as piper;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main(List<String> arguments) async {
  await sherpa_onnx.initBindingsAsync();

  final parser = ArgParser()
    ..addOption('acoustic-model', help: 'Path to the acoustic model')
    ..addOption('vocoder', help: 'Path to the vocoder model')
    ..addOption('tokens', help: 'Path to tokens.txt')
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
  if (res['acoustic-model'] == null ||
      res['vocoder'] == null ||
      res['tokens'] == null ||
      res['output-wav'] == null ||
      res['text'] == null) {
    print(parser.usage);
    exit(1);
  }
  final acousticModel = res['acoustic-model'] as String;
  final vocoder = res['vocoder'] as String;
  final tokens = res['tokens'] as String;
  final ruleFsts = res['rule-fsts'] as String;
  final ruleFars = res['rule-fars'] as String;
  final text = res['text'] as String;
  final outputWav = res['output-wav'] as String;
  var speed = double.tryParse(res['speed'] as String) ?? 1.0;
  final sid = int.tryParse(res['sid'] as String) ?? 0;

  if (speed == 0) {
    speed = 1.0;
  }

  await piper.initBindingsAsync();

  // The espeak-ng-data below feeds only the piper phonemizer; sherpa-onnx
  // ignores it. Use the copy shipped inside the model package.
  final phonemizerData = '${File(acousticModel).parent.path}/espeak-ng-data';
  final ret = piper.PiperPhonemize.initialize(phonemizerData);
  if (ret < 0) {
    print('Failed to initialize the piper phonemizer at $phonemizerData');
    exit(1);
  }

  // piper may emit empty sentences (e.g. a trailing segment after
  // punctuation); the C API requires every sentence to be non-empty.
  final sentences = piper.PiperPhonemize.phonemize(text, voice: 'en-us')
      .where((sentence) => sentence.isNotEmpty)
      .toList();

  final matcha = sherpa_onnx.OfflineTtsMatchaModelConfig(
    acousticModel: acousticModel,
    vocoder: vocoder,
    tokens: tokens,
  );

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    matcha: matcha,
    numThreads: 1,
    debug: true,
  );
  final config = sherpa_onnx.OfflineTtsConfig(
    model: modelConfig,
    maxNumSentences: 1,
    ruleFsts: ruleFsts,
    ruleFars: ruleFars,
  );

  final tts = sherpa_onnx.OfflineTts(config);
  final genConfig = sherpa_onnx.OfflineTtsGenerationConfig(
    sid: sid,
    speed: speed,
    silenceScale: 0.2,
    phonemeCodepoints: sentences,
  );
  final audio = tts.generateWithConfig(
      text: '',
      config: genConfig,
      onProgress: (Float32List samples, double progress) {
        print('${samples.length} samples received');
        return 1;
      });
  tts.free();

  sherpa_onnx.writeWave(
    filename: outputWav,
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  );
  print('Saved to $outputWav');
}
