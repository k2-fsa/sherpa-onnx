// sherpa-onnx/csrc/vocoder.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_VOCODER_H_
#define SHERPA_ONNX_CSRC_VOCODER_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

class Vocoder {
 public:
  virtual ~Vocoder() = default;

  static std::unique_ptr<Vocoder> Create(const OfflineTtsModelConfig &config);

  template <typename Manager>
  static std::unique_ptr<Vocoder> Create(Manager *mgr,
                                         const OfflineTtsModelConfig &config);

  /** @param mel A float32 tensor of shape (batch_size, feat_dim, num_frames).
   *  @return Return a float32 vector containing audio samples..
   */
  virtual std::vector<float> Run(Ort::Value mel) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VOCODER_H_
