// sherpa-onnx/csrc/offline-dolphin-attention-model.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_ATTENTION_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_ATTENTION_MODEL_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-dolphin-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

// The attention decoder branch of Dolphin, i.e., encoder.onnx + decoder.onnx.
// The encoder produces encoder_out from normalized fbank features and the
// decoder answers next-token log probabilities for a full token prefix
// (no KV cache; the prefix is re-encoded every step).
class OfflineDolphinAttentionModel {
 public:
  explicit OfflineDolphinAttentionModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineDolphinAttentionModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineDolphinAttentionModel();

  /** Run the encoder.
   *
   * @param features  A tensor of shape (1, T, 80) with normalized fbank
   *                  features. See NormalizeFeatures().
   * @param features_length  A 1-D int64 tensor of shape (1,).
   *
   * @return A tensor of shape (1, T', 512).
   */
  Ort::Value ForwardEncoder(Ort::Value features,
                            Ort::Value features_length) const;

  /** Run a single decoder step over the full prefix.
   *
   * @param encoder_out  Return value of ForwardEncoder(). It is borrowed,
   *                     i.e., not consumed.
   * @param ys           A int64 tensor of shape (1, N) with the token prefix;
   *                     ys[0] is <sos>.
   *
   * @return A tensor of shape (1, vocab_size) containing log probabilities
   *         of the next token.
   */
  Ort::Value ForwardDecoderStep(Ort::Value &encoder_out,  // NOLINT
                                Ort::Value ys) const;

  /** Apply CMVN to features in-place using the mean/invstd stored in the
   * encoder model metadata.
   */
  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const;

  int32_t VocabSize() const;

  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_DOLPHIN_ATTENTION_MODEL_H_
