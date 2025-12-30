// sherpa-onnx/csrc/rknn/offline-sense-voice-model-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_RKNN_OFFLINE_SENSE_VOICE_MODEL_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_OFFLINE_SENSE_VOICE_MODEL_RKNN_H_

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-sense-voice-model-meta-data.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelRknn {
 public:
  ~OfflineSenseVoiceModelRknn();

  explicit OfflineSenseVoiceModelRknn(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineSenseVoiceModelRknn(Manager *mgr, const OfflineModelConfig &config);

  /**
   * @param features A tensor of shape (num_frames, feature_dim)
   *                 before applying LFR.
   * @param language
   * @param text_norm
   * @returns Return a tensor of shape (num_output_frames, vocab_size)
   */
  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) const;

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_OFFLINE_SENSE_VOICE_MODEL_RKNN_H_
