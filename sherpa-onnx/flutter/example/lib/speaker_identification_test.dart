import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import 'package:path/path.dart';
import './utils.dart';

Future<void> testSpeakerID() async {
  final src =
      'assets/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx';
  final modelPath = await copyAssetFile(src: src, dst: 'model.onnx');

  final config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model: modelPath);
  final extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config: config);
  print('dim: ${extractor.dim}');
  final stream = extractor.createStream();
  print('stream.ptr: ${stream.ptr}');
  stream.free();
  print('stream.ptr: ${stream.ptr}');

  const _spk1Files = [
    'assets/sr-data/enroll/fangjun-sr-1.wav',
    'assets/sr-data/enroll/fangjun-sr-2.wav',
    'assets/sr-data/enroll/fangjun-sr-3.wav',
  ];
  final spk1Files = [];
  for (final f in _spk1Files) {
    spk1Files.add(await copyAssetFile(src: f, dst: basename(f)));
  }

  final waveData = sherpa_onnx.readWave(spk1Files[0]);
  print('num samples of ${spk1Files[0]}: ${waveData.samples.length}');

  extractor.free();
}
