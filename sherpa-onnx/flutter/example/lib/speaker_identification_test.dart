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
  final spk1Files = [];
  for (final f in _spk1Files) {
    spk1Files.add(await copyAssetFile(src: f, dst: basename(f)));
  }

  final spk1Vec = <Float32List>[];
  for (final f in spk1Files) {
    spk1Vec.add(computeEmbedding(extractor: extractor, filename: f));
  }

  print('create speaker embedding manager');

  final manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim);
  print('manager.ptr: ${manager.ptr}');

  bool ok = manager.addMulti(name: 'fangjun', embeddingList: spk1Vec);
  print('ok: $ok');

  manager.free();
  print('manager.ptr: ${manager.ptr}');

  extractor.free();
}
