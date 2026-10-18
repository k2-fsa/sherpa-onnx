// Copyright (c) 2026 LittleMouse
#ifndef SHERPA_ONNX_CSRC_AXERA_OFFLINE_TTS_ZIPVOICE_MODEL_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_OFFLINE_TTS_ZIPVOICE_MODEL_AXERA_H_
#include <memory>

#include "sherpa-onnx/csrc/offline-tts-zipvoice-model.h"
namespace sherpa_onnx {
class OfflineTtsZipvoiceModelAxera {
 public:
  explicit OfflineTtsZipvoiceModelAxera(const OfflineTtsModelConfig &config);
  ~OfflineTtsZipvoiceModelAxera();
  const OfflineTtsZipvoiceModelMetaData &GetMetaData() const;
  Ort::Value Run(Ort::Value tokens, Ort::Value prompt_tokens,
                 Ort::Value prompt_features, float speed, int32_t num_steps,
                 float t_shift, float guidance_scale) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_AXERA_OFFLINE_TTS_ZIPVOICE_MODEL_AXERA_H_
