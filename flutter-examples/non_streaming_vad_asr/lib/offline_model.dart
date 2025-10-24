import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './utils.dart';

final modelDir = 'assets';
// Remember to change `assets` in ../pubspec.yaml
// and download files to ../assets
Future<sherpa_onnx.OfflineModelConfig> getOfflineModelConfig(
    {required int type}) async {
  switch (type) {
    // whisper
    case 0: 
      return sherpa_onnx.OfflineModelConfig(
        whisper:sherpa_onnx.OfflineWhisperModelConfig(
          encoder: await copyAssetFile('$modelDir/whisper/base-encoder.onnx'),
          decoder: await copyAssetFile('$modelDir/whisper/base-decoder.onnx'),
        ),
        tokens: await copyAssetFile('$modelDir/whisper/base-tokens.txt'),
        modelType: 'whisper',
      );
    // senseVoice  
    case 1:
      return sherpa_onnx.OfflineModelConfig(
        senseVoice: sherpa_onnx.OfflineSenseVoiceModelConfig(
          model: await copyAssetFile('$modelDir/senseVoice/model.int8.onnx'), 
        ),
        tokens: await copyAssetFile('$modelDir/senseVoice/tokens.txt'),
      );
    // nemo_transducer-parakeet-tdt
    case 2:
      return sherpa_onnx.OfflineModelConfig(
        transducer: sherpa_onnx.OfflineTransducerModelConfig(
          encoder: await copyAssetFile(
              '$modelDir/nemo_transducer/encoder.int8.onnx'),
          decoder: await copyAssetFile(
              '$modelDir/nemo_transducer/decoder.int8.onnx'),
          joiner: await copyAssetFile(
              '$modelDir/nemo_transducer/joiner.int8.onnx'),
        ),
        tokens: await copyAssetFile('$modelDir/nemo_transducer/tokens.txt'),
        modelType: 'nemo_transducer',
      );
    default:
      throw ArgumentError('Unsupported type: $type');
  }
}
