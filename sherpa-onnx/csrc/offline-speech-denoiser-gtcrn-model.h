// sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_H_
#include <memory>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-model-config.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser.h"

namespace sherpa_onnx {

class OfflineSpeechDenoiserGtcrnModel {
 public:
  ~OfflineSpeechDenoiserGtcrnModel();
  explicit OfflineSpeechDenoiserGtcrnModel(
      const OfflineSpeechDenoiserModelConfig &config);

  template <typename Manager>
  OfflineSpeechDenoiserGtcrnModel(
      Manager *mgr, const OfflineSpeechDenoiserModelConfig &config);

  using States = std::vector<Ort::Value>;

  States GetInitStates() const;

  std::pair<Ort::Value, States> Run(Ort::Value x, States states) const;

  const OfflineSpeechDenoiserGtcrnModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_GTCRN_MODEL_H_
