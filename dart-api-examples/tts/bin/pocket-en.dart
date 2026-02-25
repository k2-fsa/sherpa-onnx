// Copyright (c)  2026  Xiaomi Corporation
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('lm-flow', help: 'Path to the lm flow model')
    ..addOption('lm-main', help: 'Path to the lm main model')
    ..addOption('encoder', help: 'Path to the encoder model')
    ..addOption('decoder', help: 'Path to the decoder model')
    ..addOption('text-conditioner', help: 'Path to the text conditioner model')
    ..addOption('vocab-json', help: 'Path to the vocab.json file')
    ..addOption('token-scores-json', help: 'Path to the token_scores.json file')
    ..addOption('reference-audio', help: 'Path to reference audio (wav)')
    ..addOption('text', help: 'Text to generate TTS for')
    ..addOption('output-wav', help: 'Filename to save the generated audio')
    ..addOption(
      'voice-embedding-cache-capacity',
      help: 'Voice embedding cache capacity (default: 50)',
      defaultsTo: '50',
    )
    ..addOption(
      'seed',
      help: 'Random seed for reproducibility (default: -1, random)',
      defaultsTo: '-1',
    );

  final res = parser.parse(arguments);

  if (res['lm-flow'] == null ||
      res['lm-main'] == null ||
      res['encoder'] == null ||
      res['decoder'] == null ||
      res['text-conditioner'] == null ||
      res['vocab-json'] == null ||
      res['token-scores-json'] == null ||
      res['reference-audio'] == null ||
      res['output-wav'] == null ||
      res['text'] == null) {
    print(parser.usage);
    exit(1);
  }

  final lmFlow = res['lm-flow'] as String;
  final lmMain = res['lm-main'] as String;
  final encoder = res['encoder'] as String;
  final decoder = res['decoder'] as String;
  final textConditioner = res['text-conditioner'] as String;
  final vocabJson = res['vocab-json'] as String;
  final tokenScoresJson = res['token-scores-json'] as String;
  final referenceAudioPath = res['reference-audio'] as String;
  final text = res['text'] as String;
  final outputWav = res['output-wav'] as String;
  final voiceEmbeddingCacheCapacity = int.parse(
    res['voice-embedding-cache-capacity'] as String,
  );
  final seed = int.parse(res['seed'] as String);

  // ---------------- Pocket model config ----------------
  final pocket = sherpa_onnx.OfflineTtsPocketModelConfig(
    lmFlow: lmFlow,
    lmMain: lmMain,
    encoder: encoder,
    decoder: decoder,
    textConditioner: textConditioner,
    vocabJson: vocabJson,
    tokenScoresJson: tokenScoresJson,
    voiceEmbeddingCacheCapacity: voiceEmbeddingCacheCapacity,
  );

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    pocket: pocket,
    numThreads: 1,
    debug: true,
  );

  final config = sherpa_onnx.OfflineTtsConfig(model: modelConfig);

  final tts = sherpa_onnx.OfflineTts(config);

  // ---------------- Reference audio (REQUIRED) ----------------
  final wave = sherpa_onnx.readWave(referenceAudioPath);
  if (wave.samples.isEmpty || wave.sampleRate == 0) {
    throw Exception('Failed to read reference audio: $referenceAudioPath');
  }

  final genConfig = sherpa_onnx.OfflineTtsGenerationConfig(
    sid: 0,
    speed: 1.0,
    referenceAudio: wave.samples,
    referenceSampleRate: wave.sampleRate,
    extra: {"max_reference_audio_len": 12, if (seed >= 0) "seed": seed},
  );

  // If you don't want to use a callback
  // final audio = tts.generateWithConfig(text: text, config: genConfig);

  final audio = tts.generateWithConfig(
    text: text,
    config: genConfig,
    onProgress: (samples, progress) {
      // Print progress as percentage
      print("Progress: ${(progress * 100).toStringAsFixed(2)}%");

      // Print the length of the received samples chunk
      print("Received samples length: ${samples.length}");

      // Return 1 to continue, 0 to stop generation
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
