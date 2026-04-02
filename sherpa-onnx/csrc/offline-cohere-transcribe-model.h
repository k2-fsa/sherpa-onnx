// sherpa-onnx/csrc/offline-cohere-transcribe-model.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_MODEL_H_

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineCohereTranscribeModel {
 public:
  explicit OfflineCohereTranscribeModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineCohereTranscribeModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineCohereTranscribeModel();

  /** Run the encoder model.
   *
   * @param features  A tensor of shape (N, C, T).
   *
   * @return Return a pair containing:
   *  - n_layer_cross_k: A 4-D tensor of shape
   *                     (num_layers, N, T, hidden_size)
   *  - n_layer_cross_v: A 4-D tensor of shape
   *                     (num_layers, N, T, hidden_size)
   */
  std::pair<Ort::Value, Ort::Value> ForwardEncoder(Ort::Value features) const;

  /** Run the decoder model.
   *
   * @param tokens A int64 tensor of shape (N, num_tokens)
   * @param n_layer_self_k_cache  A 5-D tensor of shape
   *                          (num_layers, N, num_heads, max_seq_len, head_dim).
   * @param n_layer_self_v_cache  A 5-D tensor of shape
   *                          (num_layers, N, num_heads, max_seq_len, head_dim).
   * @param n_layer_cross_k       A 4-D tensor of shape
   *                              (num_layers, N, T, hidden_size)
   * @param n_layer_cross_v       A 4-D tensor of shape
   *                              (num_layers, N, T, hidden_size)
   * @param offset A int64 tensor of shape (1,)
   *
   * Note that
   *  - max_seq_len is 1024 in the config.json from the original model
   *  - hidden_size is 1024
   *  - num_layers in the decoder is 8
   *  - num_heads is 8 (num_attention_heads in config.json)
   *  - head_dim is hidden_size/num_heads = 1024/8 = 128
   *  - vocab_size is 16384
   *
   * @return Return a tuple containing 3 tensors:
   *
   *  - logits A 3-D tensor of shape (N, num_tokens, vocab_size)
   *  - out_n_layer_self_k_cache Same shape as n_layer_self_k_cache
   *  - out_n_layer_self_v_cache Same shape as n_layer_self_v_cache
   */
  std::tuple<Ort::Value, Ort::Value, Ort::Value> ForwardDecoder(
      Ort::Value tokens, Ort::Value n_layer_self_k_cache,
      Ort::Value n_layer_self_v_cache, Ort::Value n_layer_cross_k,
      Ort::Value n_layer_cross_v, Ort::Value offset) const;

  /** Return the initial self kv cache in a pair
   *  - n_layer_self_k_cache A 5-D tensor of shape
   *                         (n_text_layer, N, n_audio_ctx, n_text_state).
   *  - n_layer_self_v_cache A 5-D tensor of shape
   *                         (n_text_layer, N, n_audio_ctx, n_text_state).
   */
  std::pair<Ort::Value, Ort::Value> GetInitialSelfKVCache() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;
  int32_t GetMaxSeqLen() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_MODEL_H_
