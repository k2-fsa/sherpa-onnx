// sherpa-onnx/csrc/offline-tts-vits-model.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_H_

#include <memory>
#include <string>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

class OfflineTtsVitsModel {
 public:
  ~OfflineTtsVitsModel();

  explicit OfflineTtsVitsModel(const OfflineTtsModelConfig &config);

  /** Run the model.
   *
   * @param x A int64 tensor of shape (1, num_tokens)
   * @return Return a float32 tensor containing audio samples. You can flatten
   *         it to a 1-D tensor.
   */
  Ort::Value Run(Ort::Value x);

  // Sample rate of the generated audio
  int32_t SampleRate() const;

  // true to insert a blank between each token
  bool AddBlank() const;

  std::string Punctuations() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_MODEL_H_
