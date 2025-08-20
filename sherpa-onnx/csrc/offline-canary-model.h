// sherpa-onnx/csrc/offline-canary-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_CANARY_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CANARY_MODEL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-canary-model-meta-data.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

// see
// https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/canary/test_180m_flash.py
class OfflineCanaryModel {
 public:
  explicit OfflineCanaryModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineCanaryModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineCanaryModel();

  /** Run the encoder.
   *
   * @param features  A tensor of shape (N, T, C) of dtype float32.
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int64_t.
   *
   * @return Return a vector containing:
   *  - encoder_states: A 3-D tensor of shape (N, T', encoder_dim)
   *  - encoder_len: A 1-D tensor of shape (N,) containing number
   *                        of frames in `encoder_out` before padding.
   *                        Its dtype is int64_t
   *  - enc_mask: A 2-D tensor of shape (N, T') with dtype bool
   */
  std::vector<Ort::Value> ForwardEncoder(Ort::Value features,
                                         Ort::Value features_length) const;

  /** Run the decoder model.
   *
   * @param tokens A int32 tensor of shape (N, num_tokens)
   * @param decoder_states std::vector<Ort::Value>
   * @param encoder_states Output from ForwardEncoder()
   * @param enc_mask Output from ForwardEncoder()
   *
   * @return Return a pair:
   *
   *  - logits A 3-D tensor of shape (N, num_words, vocab_size)
   *  - new_decoder_states: Can be used as input for ForwardDecoder()
   */
  std::pair<Ort::Value, std::vector<Ort::Value>> ForwardDecoder(
      Ort::Value tokens, std::vector<Ort::Value> decoder_states,
      Ort::Value encoder_states, Ort::Value enc_mask) const;

  // The return value can be used as input for ForwardDecoder()
  std::vector<Ort::Value> GetInitialDecoderStates() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

  const OfflineCanaryModelMetaData &GetModelMetadata() const;

  OfflineCanaryModelMetaData &GetModelMetadata();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CANARY_MODEL_H_
