// sherpa-onnx/csrc/axera/offline-whisper-model-axera.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_WHISPER_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_WHISPER_MODEL_AXERA_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineWhisperModelAxera {
 public:
  ~OfflineWhisperModelAxera();

  explicit OfflineWhisperModelAxera(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineWhisperModelAxera(Manager *mgr, const OfflineModelConfig &config);

  // @param features shape: (1, num_frames, feat_dim) flattened
  OfflineWhisperDecoderResult Run(std::vector<float> features) const;

  int32_t FeatureDim() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_WHISPER_MODEL_AXERA_H_