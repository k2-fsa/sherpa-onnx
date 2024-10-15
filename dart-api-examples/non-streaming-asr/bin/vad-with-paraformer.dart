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
    ..addOption('model', help: 'Path to the paraformer model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['silero-vad'] == null ||
      res['model'] == null ||
      res['tokens'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final sileroVad = res['silero-vad'] as String;
  final model = res['model'] as String;
  final tokens = res['tokens'] as String;
  final inputWav = res['input-wav'] as String;

  final paraformer = sherpa_onnx.OfflineParaformerModelConfig(
    model: model,
  );

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    paraformer: paraformer,
    tokens: tokens,
    debug: true,
    numThreads: 1,
    modelType: 'paraformer',
  );
  final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final sileroVadConfig = sherpa_onnx.SileroVadModelConfig(
    model: sileroVad,
    minSilenceDuration: 0.25,
    minSpeechDuration: 0.5,
  );

  final vadConfig = sherpa_onnx.VadModelConfig(
    sileroVad: sileroVadConfig,
    numThreads: 1,
    debug: true,
  );

  final vad = sherpa_onnx.VoiceActivityDetector(
      config: vadConfig, bufferSizeInSeconds: 10);

  final waveData = sherpa_onnx.readWave(inputWav);

  int numSamples = waveData.samples.length;
  int numIter = numSamples ~/ vadConfig.sileroVad.windowSize;

  for (int i = 0; i != numIter; ++i) {
    int start = i * vadConfig.sileroVad.windowSize;
    vad.acceptWaveform(Float32List.sublistView(
        waveData.samples, start, start + vadConfig.sileroVad.windowSize));

    while (!vad.isEmpty()) {
      final stream = recognizer.createStream();
      final segment = vad.front();
      stream.acceptWaveform(
          samples: segment.samples, sampleRate: waveData.sampleRate);
      recognizer.decode(stream);

      final result = recognizer.getResult(stream);

      final startTime = segment.start * 1.0 / waveData.sampleRate;
      final duration = segment.samples.length * 1.0 / waveData.sampleRate;
      final stopTime = startTime + duration;
      if (result.text != '') {
        print(
            '${startTime.toStringAsPrecision(4)} -- ${stopTime.toStringAsPrecision(4)}: ${result.text}');
      }

      stream.free();
      vad.pop();
    }
  }

  vad.flush();
  while (!vad.isEmpty()) {
    final stream = recognizer.createStream();
    final segment = vad.front();
    stream.acceptWaveform(
        samples: segment.samples, sampleRate: waveData.sampleRate);
    recognizer.decode(stream);

    final result = recognizer.getResult(stream);

    final startTime = segment.start * 1.0 / waveData.sampleRate;
    final duration = segment.samples.length * 1.0 / waveData.sampleRate;
    final stopTime = startTime + duration;
    if (result.text != '') {
      print(
          '${startTime.toStringAsPrecision(4)} -- ${stopTime.toStringAsPrecision(4)}: ${result.text}');
    }

    stream.free();
    vad.pop();
  }

  vad.free();
  recognizer.free();
}
