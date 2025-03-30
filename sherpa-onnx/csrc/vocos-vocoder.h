// sherpa-onnx/csrc/vocos-vocoder.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_VOCOS_VOCODER_H_
#define SHERPA_ONNX_CSRC_VOCOS_VOCODER_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/vocoder.h"

namespace sherpa_onnx {

class VocosVocoder : public Vocoder {
 public:
  ~VocosVocoder() override;

  explicit VocosVocoder(const OfflineTtsModelConfig &config);

  template <typename Manager>
  VocosVocoder(Manager *mgr, const OfflineTtsModelConfig &config);

  /** @param mel A float32 tensor of shape (batch_size, feat_dim, num_frames).
   *  @return Return a float32 tensor of shape (batch_size, num_samples).
   */
  std::vector<float> Run(Ort::Value mel) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VOCOS_VOCODER_H_
