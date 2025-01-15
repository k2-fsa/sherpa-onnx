// sherpa-onnx/csrc/offline-tts-kokoro-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsKokoroModelConfig {
  std::string model;
  std::string voices;
  std::string tokens;

  std::string data_dir;

  // speed = 1 / length_scale
  float length_scale = 1.0;

  OfflineTtsKokoroModelConfig() = default;

  OfflineTtsKokoroModelConfig(const std::string &model,
                              const std::string &voices,
                              const std::string &tokens,
                              const std::string &data_dir, float length_scale)
      : model(model),
        voices(voices),
        tokens(tokens),
        data_dir(data_dir),
        length_scale(length_scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_KOKORO_MODEL_CONFIG_H_
