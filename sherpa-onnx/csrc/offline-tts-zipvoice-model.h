// sherpa-onnx/csrc/offline-tts-zipvoice-model.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_H_

#include <memory>
#include <string>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-model-meta-data.h"

namespace sherpa_onnx {

class OfflineTtsZipvoiceModel {
 public:
  ~OfflineTtsZipvoiceModel();

  explicit OfflineTtsZipvoiceModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsZipvoiceModel(Manager *mgr, const OfflineTtsModelConfig &config);

  // Return a float32 tensor containing the mel
  // of shape (batch_size, mel_dim, num_frames)
  Ort::Value Run(Ort::Value tokens, Ort::Value prompt_tokens,
                 Ort::Value prompt_features, float speed,
                 int32_t num_steps) const;

  const OfflineTtsZipvoiceModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_H_
