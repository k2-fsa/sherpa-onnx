// Copyright (c)  2026  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('encoder', help: 'Path to the encoder model')
    ..addOption('decoder', help: 'Path to the decoder model')
    ..addOption('vocoder', help: 'Path to the vocoder model')
    ..addOption('data-dir', help: 'Path to espeak-ng-data directory')
    ..addOption('lexicon', help: 'Path to lexicon.txt')
    ..addOption('reference-audio', help: 'Path to reference audio (wav)')
    ..addOption('reference-text', help: 'Reference text for zero-shot TTS')
    ..addOption('text', help: 'Text to generate TTS for')
    ..addOption('output-wav', help: 'Filename to save the generated audio')
    ..addOption(
      'num-steps',
      help: 'Number of inference steps (default: 4)',
      defaultsTo: '4',
    );

  final res = parser.parse(arguments);

  if (res['tokens'] == null ||
      res['encoder'] == null ||
      res['decoder'] == null ||
      res['vocoder'] == null ||
      res['data-dir'] == null ||
      res['lexicon'] == null ||
      res['reference-audio'] == null ||
      res['reference-text'] == null ||
      res['output-wav'] == null ||
      res['text'] == null) {
    print(parser.usage);
    exit(1);
  }

  final tokens = res['tokens'] as String;
  final encoder = res['encoder'] as String;
  final decoder = res['decoder'] as String;
  final vocoder = res['vocoder'] as String;
  final dataDir = res['data-dir'] as String;
  final lexicon = res['lexicon'] as String;
  final referenceAudioPath = res['reference-audio'] as String;
  final referenceText = res['reference-text'] as String;
  final text = res['text'] as String;
  final outputWav = res['output-wav'] as String;
  final numSteps = int.parse(res['num-steps'] as String);

  final zipvoice = sherpa_onnx.OfflineTtsZipVoiceModelConfig(
    tokens: tokens,
    encoder: encoder,
    decoder: decoder,
    vocoder: vocoder,
    dataDir: dataDir,
    lexicon: lexicon,
  );

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    zipvoice: zipvoice,
    numThreads: 2,
    debug: true,
  );

  final config = sherpa_onnx.OfflineTtsConfig(model: modelConfig);

  final tts = sherpa_onnx.OfflineTts(config);

  final wave = sherpa_onnx.readWave(referenceAudioPath);
  if (wave.samples.isEmpty || wave.sampleRate == 0) {
    throw Exception('Failed to read reference audio: $referenceAudioPath');
  }

  final genConfig = sherpa_onnx.OfflineTtsGenerationConfig(
    speed: 1.0,
    referenceAudio: wave.samples,
    referenceSampleRate: wave.sampleRate,
    referenceText: referenceText,
    numSteps: numSteps,
    extra: {'min_char_in_sentence': 10},
  );

  final audio = tts.generateWithConfig(
    text: text,
    config: genConfig,
    onProgress: (samples, progress) {
      print('Progress: ${(progress * 100).toStringAsFixed(2)}%');
      print('Received samples length: ${samples.length}');
      return 1;
    },
  );

  tts.free();

  sherpa_onnx.writeWave(
    filename: outputWav,
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  );

  print('Saved to $outputWav');
}
