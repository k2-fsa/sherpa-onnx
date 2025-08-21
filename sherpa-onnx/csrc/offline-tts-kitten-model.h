// sherpa-onnx/csrc/offline-tts-kitten-model.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_KITTEN_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_KITTEN_MODEL_H_

#include <memory>
#include <string>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-kitten-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

class OfflineTtsKittenModel {
 public:
  ~OfflineTtsKittenModel();

  explicit OfflineTtsKittenModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsKittenModel(Manager *mgr, const OfflineTtsModelConfig &config);

  // @params x An int64 tensor of shape (1, num_tokens)
  // @return Return a float32 tensor containing the
  //         samples of shape (num_samples,)
  Ort::Value Run(Ort::Value x, int64_t sid = 0, float speed = 1.0) const;

  const OfflineTtsKittenModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_KITTEN_MODEL_H_
