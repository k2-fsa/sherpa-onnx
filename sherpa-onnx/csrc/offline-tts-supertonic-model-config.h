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
  // Directory containing ONNX models and config files
  // Expected files:
  //   - duration_predictor.onnx
  //   - text_encoder.onnx
  //   - vector_estimator.onnx
  //   - vocoder.onnx
  //   - tts.json (config file)
  //   - unicode_indexer.json
  std::string model_dir;

  // Path to voice style JSON file(s)
  // For batch inference, multiple files can be specified separated by comma
  std::string voice_style;

  // Number of denoising steps (default: 5)
  int32_t num_steps = 5;

  // Speech speed factor (default: 1.05)
  float speed = 1.05f;

  OfflineTtsSupertonicModelConfig() = default;

  OfflineTtsSupertonicModelConfig(const std::string &model_dir,
                                  const std::string &voice_style,
                                  int32_t num_steps = 5, float speed = 1.05f)
      : model_dir(model_dir),
        voice_style(voice_style),
        num_steps(num_steps),
        speed(speed) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_CONFIG_H_
