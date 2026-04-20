// sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model.h
//
// Copyright (c)  2026  Ceva Inc
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_MODEL_H_

#include <memory>
#include <utility>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-model-config.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser.h"

namespace sherpa_onnx {

class OfflineSpeechDenoiserDpdfNetModel {
 public:
  ~OfflineSpeechDenoiserDpdfNetModel();
  explicit OfflineSpeechDenoiserDpdfNetModel(
      const OfflineSpeechDenoiserModelConfig &config);

  template <typename Manager>
  OfflineSpeechDenoiserDpdfNetModel(
      Manager *mgr, const OfflineSpeechDenoiserModelConfig &config);

  Ort::Value GetInitState() const;

  std::pair<Ort::Value, Ort::Value> Run(Ort::Value x, Ort::Value state) const;

  const OfflineSpeechDenoiserDpdfNetModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEECH_DENOISER_DPDFNET_MODEL_H_
