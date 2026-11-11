// Copyright (c)  2026  Xiaomi Corporation
//
// Same as moonshine.dart, but creates the recognizer with
// OfflineRecognizer.createAsync() so the model loads in a background isolate
// and the main isolate keeps running.
import 'dart:async';
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main(List<String> arguments) async {
  await sherpa_onnx.initBindingsAsync();

  final parser = ArgParser()
    ..addOption('preprocessor',
        help: 'Path to the moonshine preprocessor model')
    ..addOption('encoder', help: 'Path to the moonshine encoder model')
    ..addOption('uncached-decoder',
        help: 'Path to moonshine uncached decoder model')
    ..addOption('cached-decoder',
        help: 'Path to moonshine cached decoder model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['preprocessor'] == null ||
      res['encoder'] == null ||
      res['uncached-decoder'] == null ||
      res['cached-decoder'] == null ||
      res['tokens'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final preprocessor = res['preprocessor'] as String;
  final encoder = res['encoder'] as String;
  final uncachedDecoder = res['uncached-decoder'] as String;
  final cachedDecoder = res['cached-decoder'] as String;
  final tokens = res['tokens'] as String;
  final inputWav = res['input-wav'] as String;

  final moonshine = sherpa_onnx.OfflineMoonshineModelConfig(
    preprocessor: preprocessor,
    encoder: encoder,
    uncachedDecoder: uncachedDecoder,
    cachedDecoder: cachedDecoder,
  );

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    moonshine: moonshine,
    tokens: tokens,
    debug: false,
    numThreads: 1,
  );
  final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);

  // The main isolate is not blocked while the model loads, so this timer
  // keeps firing.
  var ticks = 0;
  final timer = Timer.periodic(
    const Duration(milliseconds: 10),
    (_) => ticks++,
  );
  final start = DateTime.now();
  final recognizer = await sherpa_onnx.OfflineRecognizer.createAsync(config);
  timer.cancel();
  final elapsed = DateTime.now().difference(start).inMilliseconds;
  print('Model loaded in $elapsed ms, main isolate timer ticked $ticks times');

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
