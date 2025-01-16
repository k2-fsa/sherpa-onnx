// sherpa-onnx/csrc/offline-tts-matcha-model.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_H_

#include <memory>
#include <string>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-matcha-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

class OfflineTtsMatchaModel {
 public:
  ~OfflineTtsMatchaModel();

  explicit OfflineTtsMatchaModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsMatchaModel(Manager *mgr, const OfflineTtsModelConfig &config);

  // Return a float32 tensor containing the mel
  // of shape (batch_size, mel_dim, num_frames)
  Ort::Value Run(Ort::Value x, int64_t sid = 0, float speed = 1.0) const;

  const OfflineTtsMatchaModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MATCHA_MODEL_H_
