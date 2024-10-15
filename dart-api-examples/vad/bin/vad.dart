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
    ..addOption('input-wav', help: 'Path to input.wav')
    ..addOption('output-wav', help: 'Path to output.wav');

  final res = parser.parse(arguments);
  if (res['silero-vad'] == null ||
      res['input-wav'] == null ||
      res['output-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final sileroVad = res['silero-vad'] as String;
  final inputWav = res['input-wav'] as String;
  final outputWav = res['output-wav'] as String;

  final sileroVadConfig = sherpa_onnx.SileroVadModelConfig(
    model: sileroVad,
    minSilenceDuration: 0.25,
    minSpeechDuration: 0.5,
  );

  final config = sherpa_onnx.VadModelConfig(
    sileroVad: sileroVadConfig,
    numThreads: 1,
    debug: true,
  );

  final vad = sherpa_onnx.VoiceActivityDetector(
      config: config, bufferSizeInSeconds: 10);

  final waveData = sherpa_onnx.readWave(inputWav);
  if (waveData.sampleRate != 16000) {
    print('Only 16000 Hz is supported. Given: ${waveData.sampleRate}');
    exit(1);
  }

  int numSamples = waveData.samples.length;
  int numIter = numSamples ~/ config.sileroVad.windowSize;

  List<List<double>> allSamples = [];

  for (int i = 0; i != numIter; ++i) {
    int start = i * config.sileroVad.windowSize;
    vad.acceptWaveform(Float32List.sublistView(
        waveData.samples, start, start + config.sileroVad.windowSize));

    if (vad.isDetected()) {
      while (!vad.isEmpty()) {
        allSamples.add(vad.front().samples);
        vad.pop();
      }
    }
  }

  vad.flush();
  while (!vad.isEmpty()) {
    allSamples.add(vad.front().samples);
    vad.pop();
  }

  vad.free();

  final s = Float32List.fromList(allSamples.expand((x) => x).toList());
  sherpa_onnx.writeWave(
      filename: outputWav, samples: s, sampleRate: waveData.sampleRate);

  print('Saved to $outputWav');
}
