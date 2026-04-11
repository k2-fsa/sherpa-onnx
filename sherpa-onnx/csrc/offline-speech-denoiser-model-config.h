// sherpa-onnx/csrc/offline-speech-denoiser-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model-config.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineSpeechDenoiserModelConfig {
  OfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  OfflineSpeechDenoiserDpdfNetModelConfig dpdfnet;

  int32_t num_threads = 1;
  bool enable_cpu_mem_arena = true;
  bool enable_mem_pattern = true;
  bool debug = false;
  std::string provider = "cpu";

  OfflineSpeechDenoiserModelConfig() = default;

  OfflineSpeechDenoiserModelConfig(
      const OfflineSpeechDenoiserGtcrnModelConfig &gtcrn,
      const OfflineSpeechDenoiserDpdfNetModelConfig &dpdfnet,
      int32_t num_threads, bool debug, const std::string &provider,
      bool enable_cpu_mem_arena = true, bool enable_mem_pattern = true)
      : gtcrn(gtcrn),
        dpdfnet(dpdfnet),
        num_threads(num_threads),
        enable_cpu_mem_arena(enable_cpu_mem_arena),
        enable_mem_pattern(enable_mem_pattern),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_MODEL_CONFIG_H_
