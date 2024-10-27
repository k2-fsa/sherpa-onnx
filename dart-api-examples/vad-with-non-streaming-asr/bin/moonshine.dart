// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()
    ..addOption('silero-vad', help: 'Path to silero_vad.onnx')
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
  if (res['silero-vad'] == null ||
      res['preprocessor'] == null ||
      res['encoder'] == null ||
      res['uncached-decoder'] == null ||
      res['cached-decoder'] == null ||
      res['tokens'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  // create VAD
  final sileroVad = res['silero-vad'] as String;

  final sileroVadConfig = sherpa_onnx.SileroVadModelConfig(
    model: sileroVad,
    minSilenceDuration: 0.25,
    minSpeechDuration: 0.5,
    maxSpeechDuration: 5.0,
  );

  final vadConfig = sherpa_onnx.VadModelConfig(
    sileroVad: sileroVadConfig,
    numThreads: 1,
    debug: true,
  );

  final vad = sherpa_onnx.VoiceActivityDetector(
      config: vadConfig, bufferSizeInSeconds: 10);

  // create whisper recognizer
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
  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final waveData = sherpa_onnx.readWave(inputWav);
  if (waveData.sampleRate != 16000) {
    print('Only 16000 Hz is supported. Given: ${waveData.sampleRate}');
    exit(1);
  }

  int numSamples = waveData.samples.length;
  int numIter = numSamples ~/ vadConfig.sileroVad.windowSize;

  for (int i = 0; i != numIter; ++i) {
    int start = i * vadConfig.sileroVad.windowSize;
    vad.acceptWaveform(Float32List.sublistView(
        waveData.samples, start, start + vadConfig.sileroVad.windowSize));

    while (!vad.isEmpty()) {
      final samples = vad.front().samples;
      final startTime = vad.front().start.toDouble() / waveData.sampleRate;
      final endTime =
          startTime + samples.length.toDouble() / waveData.sampleRate;

      final stream = recognizer.createStream();
      stream.acceptWaveform(samples: samples, sampleRate: waveData.sampleRate);
      recognizer.decode(stream);

      final result = recognizer.getResult(stream);
      stream.free();
      print(
          '${startTime.toStringAsPrecision(5)} -- ${endTime.toStringAsPrecision(5)} : ${result.text}');

      vad.pop();
    }
  }

  vad.flush();

  while (!vad.isEmpty()) {
    final samples = vad.front().samples;
    final startTime = vad.front().start.toDouble() / waveData.sampleRate;
    final endTime = startTime + samples.length.toDouble() / waveData.sampleRate;

    final stream = recognizer.createStream();
    stream.acceptWaveform(samples: samples, sampleRate: waveData.sampleRate);
    recognizer.decode(stream);

    final result = recognizer.getResult(stream);
    stream.free();
    print(
        '${startTime.toStringAsPrecision(5)} -- ${endTime.toStringAsPrecision(5)} : ${result.text}');

    vad.pop();
  }

  vad.free();

  recognizer.free();
}
