// sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model-config.h
//
// Copyright (c)  2026  Ceva Inc
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineSpeechDenoiserDpdfNetModelConfig {
  std::string model;
  OfflineSpeechDenoiserDpdfNetModelConfig() = default;

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_MODEL_CONFIG_H_
