// Copyright (c)  2026  Xiaomi Corporation
// Model configuration for VAD.
import 'package:sherpa_onnx/sherpa_onnx.dart';

/// Select which VAD model to use (0-1).
const int selectedModelIndex = 0;

/// Model info for each selection.
const List<Map<String, String>> modelInfo = [
  {
    'name': 'Silero VAD v4',
    'file': 'silero_vad.onnx',
    'url': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx',
    'doc': 'https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html',
  },
  {
    'name': 'Ten VAD',
    'file': 'ten-vad.onnx',
    'url': 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx',
    'doc': 'https://k2-fsa.github.io/sherpa/onnx/vad/ten-vad.html',
  },
];

/// Selected model file name.
String get modelFile => modelInfo[selectedModelIndex]['file']!;

/// Download URL for the selected model.
String get modelUrl => modelInfo[selectedModelIndex]['url']!;

/// Documentation URL for the selected model.
String get modelDocUrl => modelInfo[selectedModelIndex]['doc']!;

/// Selected model name.
String get modelName => modelInfo[selectedModelIndex]['name']!;

/// Default VAD config for the selected model.
VadModelConfig get defaultVadConfig {
  if (selectedModelIndex == 1) {
    // Ten VAD
    return VadModelConfig(
      tenVad: TenVadModelConfig(
        model: modelFile,
        threshold: 0.25,
        minSilenceDuration: 0.25,
        minSpeechDuration: 0.25,
        windowSize: 256,
        maxSpeechDuration: 5.0,
      ),
      sampleRate: 16000,
      numThreads: 1,
      debug: false,
    );
  }
  // Silero VAD v4 (default)
  return VadModelConfig(
    sileroVad: SileroVadModelConfig(
      model: modelFile,
      threshold: 0.1,
      minSilenceDuration: 0.5,
      minSpeechDuration: 0.25,
      windowSize: 512,
      maxSpeechDuration: 12.0,
    ),
    sampleRate: 16000,
    numThreads: 1,
    debug: false,
  );
}

/// Window size for the selected model.
int get windowSize => selectedModelIndex == 1 ? 256 : 512;
