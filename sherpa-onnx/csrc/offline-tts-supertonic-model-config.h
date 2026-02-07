// sherpa-onnx/csrc/offline-tts-supertonic-model-config.h
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsSupertonicModelConfig {
  // Individual model file paths
  std::string duration_predictor;
  std::string text_encoder;
  std::string vector_estimator;
  std::string vocoder;

  // Directory containing config files (tts.json and unicode_indexer.json)
  std::string model_dir;

  // Path to voice style .bin file(s);
  std::string voice_style;

  OfflineTtsSupertonicModelConfig() = default;

  OfflineTtsSupertonicModelConfig(const std::string &duration_predictor,
                                  const std::string &text_encoder,
                                  const std::string &vector_estimator,
                                  const std::string &vocoder,
                                  const std::string &model_dir,
                                  const std::string &voice_style)
      : duration_predictor(duration_predictor),
        text_encoder(text_encoder),
        vector_estimator(vector_estimator),
        vocoder(vocoder),
        model_dir(model_dir),
        voice_style(voice_style) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_CONFIG_H_
