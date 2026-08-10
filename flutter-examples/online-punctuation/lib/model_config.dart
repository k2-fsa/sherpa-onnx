// Copyright (c)  2026  Xiaomi Corporation
// Shared model selection — edit this file to change the punctuation model.
// Both native (model.dart) and web (model_web.dart) use this.
//
import 'package:sherpa_onnx/sherpa_onnx.dart';

/// Model directory name.
const String modelDir = 'sherpa-onnx-online-punct-en-2024-08-06';

/// Model file name.
const String modelFile = 'model.int8.onnx';

/// BPE vocab file name.
const String bpeVocabFile = 'bpe.vocab';

/// Download URL for the model.
const String modelUrl =
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/$modelDir.tar.bz2';

/// Documentation URL.
const String modelDocUrl =
    'https://k2-fsa.github.io/sherpa/onnx/punctuation/pretrained_models.html#sherpa-onnx-online-punct-en-2024-08-06';

/// Online punctuation config.
final OnlinePunctuationConfig punctConfig = OnlinePunctuationConfig(
  model: OnlinePunctuationModelConfig(
    cnnBiLstm: '$modelDir/$modelFile',
    bpeVocab: '$modelDir/$bpeVocabFile',
    numThreads: 1,
    debug: true,
  ),
);
