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
    ..addOption('encoder', help: 'Path to the encoder model')
    ..addOption('decoder', help: 'Path to decoder model')
    ..addOption('joiner', help: 'Path to joiner model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);

  if (res['silero-vad'] == null ||
      res['encoder'] == null ||
      res['decoder'] == null ||
      res['joiner'] == null ||
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
  );

  final vadConfig = sherpa_onnx.VadModelConfig(
    sileroVad: sileroVadConfig,
    numThreads: 1,
    debug: true,
  );

  final vad = sherpa_onnx.VoiceActivityDetector(
      config: vadConfig, bufferSizeInSeconds: 10);

  // create zipformer transducer recognizer
  final encoder = res['encoder'] as String;
  final decoder = res['decoder'] as String;
  final joiner = res['joiner'] as String;
  final tokens = res['tokens'] as String;
  final inputWav = res['input-wav'] as String;

  final transducer = sherpa_onnx.OfflineTransducerModelConfig(
    encoder: encoder,
    decoder: decoder,
    joiner: joiner,
  );

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    transducer: transducer,
    tokens: tokens,
    debug: true,
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

    if (vad.isDetected()) {
      while (!vad.isEmpty()) {
        final samples = vad.front().samples;
        final startTime = vad.front().start.toDouble() / waveData.sampleRate;
        final endTime =
            startTime + samples.length.toDouble() / waveData.sampleRate;

        final stream = recognizer.createStream();
        stream.acceptWaveform(
            samples: samples, sampleRate: waveData.sampleRate);
        recognizer.decode(stream);

        final result = recognizer.getResult(stream);
        stream.free();
        print(
            '${startTime.toStringAsPrecision(5)} -- ${endTime.toStringAsPrecision(5)} : ${result.text}');

        vad.pop();
      }
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
