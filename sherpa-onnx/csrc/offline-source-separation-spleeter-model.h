// sherpa-onnx/csrc/offline-source-separation-spleeter-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_MODEL_H_
#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-source-separation-model-config.h"
#include "sherpa-onnx/csrc/offline-source-separation-spleeter-model-meta-data.h"

namespace sherpa_onnx {

class OfflineSourceSeparationSpleeterModel {
 public:
  ~OfflineSourceSeparationSpleeterModel();

  explicit OfflineSourceSeparationSpleeterModel(
      const OfflineSourceSeparationModelConfig &config);

  template <typename Manager>
  OfflineSourceSeparationSpleeterModel(
      Manager *mgr, const OfflineSourceSeparationModelConfig &config);

  Ort::Value RunVocals(Ort::Value x) const;
  Ort::Value RunAccompaniment(Ort::Value x) const;

  const OfflineSourceSeparationSpleeterModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_MODEL_H_
