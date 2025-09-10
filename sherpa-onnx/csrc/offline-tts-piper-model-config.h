// sherpa-onnx/csrc/offline-tts-piper-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsPiperModelConfig {
  std::string model;
  std::string model_config_file;
  std::string data_dir;

  OfflineTtsPiperModelConfig() = default;

  OfflineTtsPiperModelConfig(const std::string &model,
                            const std::string &model_config_file, 
                            const std::string &data_dir)
      : model(model), 
        model_config_file(model_config_file),
        data_dir(data_dir) {}

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_PIPER_MODEL_CONFIG_H_