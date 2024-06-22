import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './utils.dart';

// Remember to change `assets` in ../pubspec.yaml
// and download files to ../assets
Future<sherpa_onnx.OnlineModelConfig> getOnlineModelConfig(
    {required int type}) async {
  switch (type) {
    case 1:
      final modelDir =
          "assets/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20";
      return sherpa_onnx.OnlineModelConfig(
        transducer: sherpa_onnx.OnlineTransducerModelConfig(
          encoder:
              await copyAssetFile("$modelDir/encoder-epoch-99-avg-1.int8.onnx"),
          decoder: await copyAssetFile("$modelDir/decoder-epoch-99-avg-1.onnx"),
          joiner:
              await copyAssetFile("$modelDir/joiner-epoch-99-avg-1.int8.onnx"),
        ),
        tokens: await copyAssetFile("$modelDir/tokens.txt"),
        modelType: 'zipformer',
      );
    default:
      throw ArgumentError("Unsupported type: $type");
  }
}
