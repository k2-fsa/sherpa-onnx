// Copyright (c)  2024  Xiaomi Corporation
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import 'dart:typed_data';
import 'package:path/path.dart';
import './utils.dart';

Float32List computeEmbedding(
    {required sherpa_onnx.SpeakerEmbeddingExtractor extractor,
    required String filename}) {
  final stream = extractor.createStream();
  final waveData = sherpa_onnx.readWave(filename);

  stream.acceptWaveform(
      samples: waveData.samples, sampleRate: waveData.sampleRate);

  stream.inputFinished();

  final embedding = extractor.compute(stream);

  stream.free();

  return embedding;
}

Future<void> testSpeakerID() async {
  final src =
      'assets/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx';
  final modelPath = await copyAssetFile(src: src, dst: 'model.onnx');

  final config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model: modelPath);
  final extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config: config);

  const _spk1Files = [
    'assets/sr-data/enroll/fangjun-sr-1.wav',
    'assets/sr-data/enroll/fangjun-sr-2.wav',
    'assets/sr-data/enroll/fangjun-sr-3.wav',
  ];
  final spk1Files = <String>[];
  for (final f in _spk1Files) {
    spk1Files.add(await copyAssetFile(src: f, dst: basename(f)));
  }

  final spk1Vec = <Float32List>[];
  for (final f in spk1Files) {
    spk1Vec.add(computeEmbedding(extractor: extractor, filename: f));
  }

  const _spk2Files = [
    'assets/sr-data/enroll/leijun-sr-1.wav',
    'assets/sr-data/enroll/leijun-sr-2.wav',
  ];
  final spk2Files = <String>[];
  for (final f in _spk2Files) {
    spk2Files.add(await copyAssetFile(src: f, dst: basename(f)));
  }

  final spk2Vec = <Float32List>[];
  for (final f in spk2Files) {
    spk2Vec.add(computeEmbedding(extractor: extractor, filename: f));
  }

  final manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim);
  assert(manager.numSpeakers == 0, '${manager.numSpeakers}');

  bool ok = manager.addMulti(name: 'fangjun', embeddingList: spk1Vec);
  assert(ok, "Failed to add fangjun");
  assert(manager.numSpeakers == 1, '${manager.numSpeakers}');

  ok = manager.addMulti(name: 'leijun', embeddingList: spk2Vec);
  assert(ok, "Failed to add leijun");
  assert(manager.numSpeakers == 2, '${manager.numSpeakers}');

  bool found = manager.contains('fangjun');
  assert(found, 'Failed to find fangjun');

  found = manager.contains('leijun');
  assert(found, 'Failed to find leijun');

  print('---All speakers---');

  print(manager.allSpeakerNames);

  print('------------');

  const _testFiles = [
    'assets/sr-data/test/fangjun-test-sr-1.wav',
    'assets/sr-data/test/leijun-test-sr-1.wav',
    'assets/sr-data/test/liudehua-test-sr-1.wav',
  ];

  final testFiles = <String>[];
  for (final f in _testFiles) {
    testFiles.add(await copyAssetFile(src: f, dst: basename(f)));
  }

  const threshold = 0.6;

  for (final f in testFiles) {
    final embedding = computeEmbedding(extractor: extractor, filename: f);

    var name = manager.search(embedding: embedding, threshold: threshold);
    if (name == '') {
      name = '<Unknown>';
    }
    print('${f}: ${name}');
  }

  ok = manager.verify(
      name: 'fangjun',
      embedding: computeEmbedding(extractor: extractor, filename: testFiles[0]),
      threshold: threshold);
  assert(ok, 'Failed to verify fangjun using ${testFiles[0]}');

  ok = manager.remove('fangjun');
  assert(ok, 'Failed to remove fangjun');
  assert(manager.numSpeakers == 1, '${manager.numSpeakers}');

  found = manager.contains('fangjun');
  assert(!found, 'Still found fangjun!');

  ok = manager.verify(
      name: 'fangjun',
      embedding: computeEmbedding(extractor: extractor, filename: testFiles[0]),
      threshold: threshold);
  assert(!ok, '${testFiles[0]}');

  manager.free();
  extractor.free();
}
