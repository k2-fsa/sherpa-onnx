// sherpa-onnx/csrc/offline-sense-voice-model.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SENSE_VOICE_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SENSE_VOICE_MODEL_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-sense-voice-model-meta-data.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModel {
 public:
  explicit OfflineSenseVoiceModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineSenseVoiceModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineSenseVoiceModel();

  /** Run the forward method of the model.
   *
   * @param features  A tensor of shape (N, T, C). It is changed in-place.
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int32_t.
   * @param language A 1-D tensor of shape (N,) with dtype int32_t
   * @param text_norm A 1-D tensor of shape (N,) with dtype int32_t
   *
   * @return Return logits of shape (N, T, C) with dtype float
   *
   * Note: The subsampling factor is 1 for SenseVoice, so there is
   *       no need to output logits_length.
   */
  Ort::Value Forward(Ort::Value features, Ort::Value features_length,
                     Ort::Value language, Ort::Value text_norm) const;

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SENSE_VOICE_MODEL_H_
