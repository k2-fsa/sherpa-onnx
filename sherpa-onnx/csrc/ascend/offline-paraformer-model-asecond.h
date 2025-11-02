// sherpa-onnx/csrc/ascend/offline-paraformer-model-ascend.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ASCEND_OFFLINE_PARAFORMER_MODEL_ASCEND_H_
#define SHERPA_ONNX_CSRC_ASCEND_OFFLINE_PARAFORMER_MODEL_ASCEND_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineParaformerModelAscend {
 public:
  ~OfflineParaformerModelAscend();

  explicit OfflineParaformerModelAscend(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineParaformerModelAscend(Manager *mgr, const OfflineModelConfig &config);

  /**
   * @param features A tensor of shape (num_frames, feature_dim)
   *                 before applying LFR.
   * @returns Return a tensor of shape (num_output_frames, encoder_dim)
   */
  std::vector<float> Run(std::vector<float> features) const;

#if 0

  /**
   * @param encoder_out A tensor of shape (num_frames, encoder_dim)
   * @returns Return a tensor of shape (num_frames, encoder_dim)
   */
  std::vector<float> RunPredictor(const std::vector<float> &encoder_out) const;

  /**
   * @param encoder_out A tensor of shape (num_frames, encoder_dim)
   * @param acoustic_embedding A tensor of shape (num_tokens, encoder_dim)
   * @returns Return a tensor of shape (num_tokens, vocab_size)
   */
  std::vector<float> RunDecoder(std::vector<float> encoder_out,
                                std::vector<float> acoustic_embedding) const;

  int32_t GetEncoderOutputDim() const;

  int32_t GetVocabSize() const;
#endif

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_OFFLINE_PARAFORMER_MODEL_ASCEND_H_
