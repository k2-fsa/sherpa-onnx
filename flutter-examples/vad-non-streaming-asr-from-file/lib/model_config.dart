// Copyright (c)  2026  Xiaomi Corporation
//
// Model configuration for VAD + Non-streaming ASR.
//
// Change `selectedModelIndex` to switch ASR models:
//   0 = Zipformer CTC (Chinese)
//   1 = SenseVoice (Chinese/English/Japanese/Korean/Cantonese)
//   2 = Whisper tiny.en (English)
//   3 = NeMo Parakeet TDT v2 (English)
//   4 = NeMo Parakeet TDT v3 (25 European languages)
//   5 = Moonshine tiny en (English)
//   6 = Qwen3 ASR 0.6B (multilingual)
//   7 = FunASR Nano (Chinese/English/Japanese)
//   8 = FireRed ASR CTC v2 (Chinese/English)

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

// === Change this to switch ASR model ===
// IMPORTANT: When adding a new model here, also update:
//   1. worker_web.dart — add a case in _buildAsrConfigForWeb() for the JS config
//   2. assets/ — copy the model files into the assets directory
//   3. Run: python3 generate-asset-list.py  (updates pubspec.yaml automatically)
const int selectedModelIndex = 0;

// --- VAD model (always Silero VAD) ---

const vadModelFile = 'silero_vad.onnx';
const vadModelUrl =
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx';

sherpa_onnx.VadModelConfig get defaultVadConfig =>
    sherpa_onnx.VadModelConfig(
      sileroVad: sherpa_onnx.SileroVadModelConfig(
        model: vadModelFile,
        threshold: 0.1,
        minSilenceDuration: 0.5,
        minSpeechDuration: 0.25,
        maxSpeechDuration: 5.0,
      ),
      numThreads: 1,
      debug: true,
    );

// --- ASR models ---

class AsrModelInfo {
  final String name;
  final String modelUrl;
  final String docUrl;
  final List<String> assetFiles; // files to download/copy to assets/

  const AsrModelInfo({
    required this.name,
    required this.modelUrl,
    required this.docUrl,
    required this.assetFiles,
  });
}

const List<AsrModelInfo> asrModels = [
  // 0: Zipformer CTC (Chinese)
  AsrModelInfo(
    name: 'Zipformer CTC (Chinese)',
    modelUrl:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2',
    docUrl:
        'https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html',
    assetFiles: [
      'sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx',
      'sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt',
    ],
  ),
  // 1: SenseVoice (Chinese/English/Japanese/Korean/Cantonese)
  AsrModelInfo(
    name: 'SenseVoice',
    modelUrl:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2',
    docUrl:
        'https://k2-fsa.github.io/sherpa/onnx/sense-voice/',
    assetFiles: [
      'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx',
      'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt',
    ],
  ),
  // 2: Whisper tiny.en (English)
  AsrModelInfo(
    name: 'Whisper tiny.en',
    modelUrl:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2',
    docUrl:
        'https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html',
    assetFiles: [
      'sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx',
      'sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx',
      'sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt',
    ],
  ),
  // 3: NeMo Parakeet TDT v2 (English, transducer)
  AsrModelInfo(
    name: 'NeMo Parakeet TDT v2 (English)',
    modelUrl:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2',
    docUrl:
        'https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html#sherpa-onnx-nemo-parakeet-tdt-0-6b-v2-int8-english',
    assetFiles: [
      'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx',
      'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx',
      'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx',
      'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt',
    ],
  ),
  // 4: NeMo Parakeet TDT v3 (25 European languages, transducer)
  AsrModelInfo(
    name: 'NeMo Parakeet TDT v3 (25 langs)',
    modelUrl:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2',
    docUrl:
        'https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html#sherpa-onnx-nemo-parakeet-tdt-0-6b-v3-int8-25-european-languages',
    assetFiles: [
      'sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx',
      'sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx',
      'sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx',
      'sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt',
    ],
  ),
  // 5: Moonshine tiny en (English)
  AsrModelInfo(
    name: 'Moonshine tiny en',
    modelUrl:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2',
    docUrl:
        'https://k2-fsa.github.io/sherpa/onnx/moonshine/models-v2.html#sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27-english',
    assetFiles: [
      'sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort',
      'sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort',
      'sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt',
    ],
  ),
  // 6: Qwen3 ASR 0.6B (multilingual)
  AsrModelInfo(
    name: 'Qwen3 ASR 0.6B',
    modelUrl:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2',
    docUrl:
        'https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html#sherpa-onnx-qwen3-asr-0-6b-int8-2026-03-25',
    assetFiles: [
      'sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx',
      'sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx',
      'sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx',
      'sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer/tokenizer.json',
    ],
  ),
  // 7: FunASR Nano (Chinese/English/Japanese)
  AsrModelInfo(
    name: 'FunASR Nano',
    modelUrl:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2',
    docUrl:
        'https://k2-fsa.github.io/sherpa/onnx/funasr-nano/pretrained.html#sherpa-onnx-funasr-nano-int8-2025-12-30-chinese-english-japanese',
    assetFiles: [
      'sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx',
      'sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx',
      'sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx',
      'sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B/tokenizer.json',
    ],
  ),
  // 8: FireRed ASR CTC v2 (Chinese/English)
  AsrModelInfo(
    name: 'FireRed ASR CTC v2',
    modelUrl:
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2',
    docUrl:
        'https://k2-fsa.github.io/sherpa/onnx/FireRedAsr/pretrained.html#sherpa-onnx-fire-red-asr2-ctc-zh-en-int8-2026-02-25-v2-ctc-chinese-english-20',
    assetFiles: [
      'sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx',
      'sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt',
    ],
  ),
];

AsrModelInfo get selectedAsrModel => asrModels[selectedModelIndex];

/// Build the ASR recognizer config for the selected model.
/// Paths should be absolute (resolved by model.dart on native).
sherpa_onnx.OfflineRecognizerConfig buildAsrConfig({
  required String modelDir,
}) {
  switch (selectedModelIndex) {
    case 0: // Zipformer CTC
      return sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          zipformerCtc: sherpa_onnx.OfflineZipformerCtcModelConfig(
            model: '$modelDir/model.int8.onnx',
          ),
          tokens: '$modelDir/tokens.txt',
          debug: true,
          numThreads: 2,
        ),
      );
    case 1: // SenseVoice
      return sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          senseVoice: sherpa_onnx.OfflineSenseVoiceModelConfig(
            model: '$modelDir/model.int8.onnx',
            language: 'auto',
            useInverseTextNormalization: false,
          ),
          tokens: '$modelDir/tokens.txt',
          debug: true,
          numThreads: 2,
        ),
      );
    case 2: // Whisper
      return sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          whisper: sherpa_onnx.OfflineWhisperModelConfig(
            encoder: '$modelDir/tiny.en-encoder.int8.onnx',
            decoder: '$modelDir/tiny.en-decoder.int8.onnx',
          ),
          tokens: '$modelDir/tiny.en-tokens.txt',
          modelType: 'whisper',
          debug: false,
          numThreads: 2,
        ),
      );
    case 3: // NeMo Parakeet (English, transducer)
    case 4: // NeMo Parakeet TDT v3 (transducer)
      return sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          transducer: sherpa_onnx.OfflineTransducerModelConfig(
            encoder: '$modelDir/encoder.int8.onnx',
            decoder: '$modelDir/decoder.int8.onnx',
            joiner: '$modelDir/joiner.int8.onnx',
          ),
          tokens: '$modelDir/tokens.txt',
          modelType: 'nemo_transducer',
          debug: true,
          numThreads: 2,
        ),
      );
    case 5: // Moonshine tiny en
      return sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          moonshine: sherpa_onnx.OfflineMoonshineModelConfig(
            encoder: '$modelDir/encoder_model.ort',
            mergedDecoder: '$modelDir/decoder_model_merged.ort',
          ),
          tokens: '$modelDir/tokens.txt',
          debug: true,
          numThreads: 2,
        ),
      );
    case 6: // Qwen3 ASR
      return sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          qwen3Asr: sherpa_onnx.OfflineQwen3AsrModelConfig(
            convFrontend: '$modelDir/conv_frontend.onnx',
            encoder: '$modelDir/encoder.int8.onnx',
            decoder: '$modelDir/decoder.int8.onnx',
            tokenizer: '$modelDir/tokenizer/tokenizer.json',
          ),
          tokens: '',
          debug: true,
          numThreads: 2,
        ),
      );
    case 7: // FunASR Nano
      return sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          funasrNano: sherpa_onnx.OfflineFunAsrNanoModelConfig(
            encoderAdaptor: '$modelDir/encoder_adaptor.int8.onnx',
            llm: '$modelDir/llm.int8.onnx',
            embedding: '$modelDir/embedding.int8.onnx',
            tokenizer: '$modelDir/Qwen3-0.6B/tokenizer.json',
          ),
          tokens: '',
          debug: true,
          numThreads: 2,
        ),
      );
    case 8: // FireRed ASR CTC
      return sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          fireRedAsrCtc: sherpa_onnx.OfflineFireRedAsrCtcModelConfig(
            model: '$modelDir/model.int8.onnx',
          ),
          tokens: '$modelDir/tokens.txt',
          debug: true,
          numThreads: 2,
        ),
      );
    default:
      throw ArgumentError('Unknown model index: $selectedModelIndex');
  }
}
