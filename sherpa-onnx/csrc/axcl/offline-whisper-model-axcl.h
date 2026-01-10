// sherpa-onnx/csrc/axcl/offline-whisper-model-axcl.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AXCL_OFFLINE_WHISPER_MODEL_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_OFFLINE_WHISPER_MODEL_AXCL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineWhisperModelAxcl {
 public:
  ~OfflineWhisperModelAxcl();

  explicit OfflineWhisperModelAxcl(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineWhisperModelAxcl(Manager *mgr, const OfflineModelConfig &config);

  /**
   * @param features A tensor of shape (1, num_frames, feat_dim)
   */
  OfflineWhisperDecoderResult Run(std::vector<float> features) const;

  int32_t FeatureDim() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_OFFLINE_WHISPER_MODEL_AXCL_H_
