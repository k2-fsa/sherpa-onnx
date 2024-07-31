// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './init.dart';

Float32List computeEmbedding(
    {required sherpa_onnx.SpeakerEmbeddingExtractor extractor,
    required String filename}) {
  final waveData = sherpa_onnx.readWave(filename);
  final stream = extractor.createStream();

  stream.acceptWaveform(
    samples: waveData.samples,
    sampleRate: waveData.sampleRate,
  );

  stream.inputFinished();

  final embedding = extractor.compute(stream);

  stream.free();

  return embedding;
}

void main(List<String> arguments) async {
  await initSherpaOnnx();

  final parser = ArgParser()..addOption('model', help: 'Path to model.onnx');

  final res = parser.parse(arguments);
  if (res['model'] == null) {
    print(parser.usage);
    exit(1);
  }

  final model = res['model'] as String;
  /*
     Please download test data by yourself

  curl -SL -o sr-data.tar.gz https://github.com/csukuangfj/sr-data/archive/refs/tags/v1.0.0.tar.gz
  tar xvf sr-data.tar.gz
  mv sr-data-1.0.0 sr-data
  */

  final config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model: model,
    numThreads: 1,
    debug: true,
    provider: 'cpu',
  );
  final extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config: config);

  final manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim);

  final spk1Files = [
    "./sr-data/enroll/fangjun-sr-1.wav",
    "./sr-data/enroll/fangjun-sr-2.wav",
    "./sr-data/enroll/fangjun-sr-3.wav",
  ];

  final spk1Vec = <Float32List>[];
  for (final f in spk1Files) {
    final embedding = computeEmbedding(extractor: extractor, filename: f);
    spk1Vec.add(embedding);
  }

  final spk2Files = [
    "./sr-data/enroll/leijun-sr-1.wav",
    "./sr-data/enroll/leijun-sr-2.wav",
  ];

  final spk2Vec = <Float32List>[];
  for (final f in spk2Files) {
    final embedding = computeEmbedding(extractor: extractor, filename: f);
    spk2Vec.add(embedding);
  }

  if (!manager.addMulti(name: "fangjun", embeddingList: spk1Vec)) {
    // Note you should free extractor and manager in your app to avoid memory leak
    print("Failed to register fangjun");
    return;
  }

  if (!manager.addMulti(name: "leijun", embeddingList: spk2Vec)) {
    print("Failed to register leijun");
    return;
  }

  if (manager.numSpeakers != 2) {
    print("There should be two speakers");
    return;
  }

  if (!manager.contains("fangjun")) {
    print("It should contain the speaker fangjun");
    return;
  }

  if (!manager.contains("leijun")) {
    print("It should contain the speaker leijun");
    return;
  }

  print("---All speakers---");
  final allSpeakers = manager.allSpeakerNames;
  for (final s in allSpeakers) {
    print(s);
  }
  print("------------");

  final testFiles = [
    "./sr-data/test/fangjun-test-sr-1.wav",
    "./sr-data/test/leijun-test-sr-1.wav",
    "./sr-data/test/liudehua-test-sr-1.wav",
  ];

  final threshold = 0.6;
  for (final file in testFiles) {
    final embedding = computeEmbedding(extractor: extractor, filename: file);

    var name = manager.search(embedding: embedding, threshold: threshold);
    if (name == '') {
      name = "<Unknown>";
    }
    print("$file: $name");
  }

  if (!manager.verify(
      name: "fangjun",
      embedding: computeEmbedding(extractor: extractor, filename: testFiles[0]),
      threshold: threshold)) {
    print("{$testFiles[0]} should match fangjun!");
    return;
  }

  if (!manager.remove("fangjun")) {
    print("Failed to remove fangjun");
    return;
  }

  if (manager.verify(
      name: "fangjun",
      embedding: computeEmbedding(extractor: extractor, filename: testFiles[0]),
      threshold: threshold)) {
    print("${testFiles[0]} should match no one!");
    return;
  }

  if (manager.numSpeakers != 1) {
    print("There should only 1 speaker left.");
    return;
  }

  extractor.free();
  manager.free();
}
