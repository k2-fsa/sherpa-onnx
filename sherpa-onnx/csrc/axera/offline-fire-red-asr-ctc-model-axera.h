// sherpa-onnx/csrc/axera/offline-fire-red-asr-ctc-model-axera.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_FIRE_RED_ASR_CTC_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_FIRE_RED_ASR_CTC_MODEL_AXERA_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-ctc-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineFireRedAsrCtcModelAxera : public OfflineCtcModel {
 public:
  explicit OfflineFireRedAsrCtcModelAxera(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineFireRedAsrCtcModelAxera(Manager *mgr,
                                 const OfflineModelConfig &config);

  ~OfflineFireRedAsrCtcModelAxera() override;

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) override;

  int32_t VocabSize() const override;

  int32_t SubsamplingFactor() const override;

  OrtAllocator *Allocator() const override;

  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_FIRE_RED_ASR_CTC_MODEL_AXERA_H_
