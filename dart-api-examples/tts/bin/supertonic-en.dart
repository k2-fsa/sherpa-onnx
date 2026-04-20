// Copyright (c)  2026  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('duration-predictor',
        help: 'Path to the duration predictor model')
    ..addOption('text-encoder', help: 'Path to the text encoder model')
    ..addOption('vector-estimator',
        help: 'Path to the vector estimator model')
    ..addOption('vocoder', help: 'Path to the vocoder model')
    ..addOption('tts-json', help: 'Path to tts.json')
    ..addOption('unicode-indexer', help: 'Path to unicode_indexer.bin')
    ..addOption('voice-style', help: 'Path to voice.bin')
    ..addOption('sid', help: 'Speaker ID (default: 6)', defaultsTo: '6')
    ..addOption('speed', help: 'Speed (default: 1.25)', defaultsTo: '1.25')
    ..addOption('num-steps',
        help: 'Number of steps (default: 5)', defaultsTo: '5')
    ..addOption('text', help: 'Text to generate TTS for')
    ..addOption('output-wav', help: 'Filename to save the generated audio');

  final res = parser.parse(arguments);

  if (res['duration-predictor'] == null ||
      res['text-encoder'] == null ||
      res['vector-estimator'] == null ||
      res['vocoder'] == null ||
      res['tts-json'] == null ||
      res['unicode-indexer'] == null ||
      res['voice-style'] == null ||
      res['output-wav'] == null ||
      res['text'] == null) {
    print(parser.usage);
    exit(1);
  }

  final durationPredictor = res['duration-predictor'] as String;
  final textEncoder = res['text-encoder'] as String;
  final vectorEstimator = res['vector-estimator'] as String;
  final vocoder = res['vocoder'] as String;
  final ttsJson = res['tts-json'] as String;
  final unicodeIndexer = res['unicode-indexer'] as String;
  final voiceStyle = res['voice-style'] as String;
  final sid = int.parse(res['sid'] as String);
  final speed = double.parse(res['speed'] as String);
  final numSteps = int.parse(res['num-steps'] as String);
  final text = res['text'] as String;
  final outputWav = res['output-wav'] as String;

  final supertonic = sherpa_onnx.OfflineTtsSupertonicModelConfig(
    durationPredictor: durationPredictor,
    textEncoder: textEncoder,
    vectorEstimator: vectorEstimator,
    vocoder: vocoder,
    ttsJson: ttsJson,
    unicodeIndexer: unicodeIndexer,
    voiceStyle: voiceStyle,
  );

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    supertonic: supertonic,
    numThreads: 2,
    debug: true,
  );

  final config = sherpa_onnx.OfflineTtsConfig(model: modelConfig);

  final tts = sherpa_onnx.OfflineTts(config);

  final genConfig = sherpa_onnx.OfflineTtsGenerationConfig(
    sid: sid,
    speed: speed,
    extra: {'lang': 'en', 'num_steps': numSteps},
  );

  final audio = tts.generateWithConfig(
    text: text,
    config: genConfig,
    onProgress: (samples, progress) {
      print('Progress: ${(progress * 100).toStringAsFixed(2)}%');
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
