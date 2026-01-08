// sherpa-onnx/csrc/ascend/offline-whisper-model-ascend.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ASCEND_OFFLINE_WHISPER_MODEL_ASCEND_H_
#define SHERPA_ONNX_CSRC_ASCEND_OFFLINE_WHISPER_MODEL_ASCEND_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineWhisperModelAscend {
 public:
  ~OfflineWhisperModelAscend();

  explicit OfflineWhisperModelAscend(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineWhisperModelAscend(Manager *mgr, const OfflineModelConfig &config);

  /**
   * @param features A tensor of shape (1, feat_dim, num_frames)
   * @returns Return a list of token IDs.
   */
  std::vector<int32_t> Run(std::vector<float> features) const;

  int32_t FeatureDim() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_OFFLINE_WHISPER_MODEL_ASCEND_H_
