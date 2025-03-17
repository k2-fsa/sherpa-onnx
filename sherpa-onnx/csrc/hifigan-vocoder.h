// sherpa-onnx/csrc/hifigan-vocoder.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_HIFIGAN_VOCODER_H_
#define SHERPA_ONNX_CSRC_HIFIGAN_VOCODER_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/vocoder.h"

namespace sherpa_onnx {

class HifiganVocoder : public Vocoder {
 public:
  ~HifiganVocoder() override;

  HifiganVocoder(int32_t num_threads, const std::string &provider,
                 const std::string &model);

  template <typename Manager>
  HifiganVocoder(Manager *mgr, int32_t num_threads, const std::string &provider,
                 const std::string &model);

  /** @param mel A float32 tensor of shape (batch_size, feat_dim, num_frames).
   *  @return Return a float32 tensor of shape (batch_size, num_samples).
   */
  std::vector<float> Run(Ort::Value mel) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_HIFIGAN_VOCODER_H_
