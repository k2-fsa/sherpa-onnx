// sherpa-onnx/csrc/offline-funasr-nano-model.h
//
// Copyright (c)  2025  zengyw

#ifndef SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-funasr-nano-model-config.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

class OfflineFunASRNanoModel {
 public:
  explicit OfflineFunASRNanoModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineFunASRNanoModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineFunASRNanoModel();

  /** Run the encoder+adaptor model.
   *
   * @param features  A tensor of shape (N, T, C). Audio features.
   * @return Return embeddings of shape (N, T', hidden_size)
   */
  Ort::Value ForwardEncoderAdaptor(Ort::Value features);

  /** Run the unified LLM model (KV cache mode).
   *
   * @param inputs_embeds  A tensor of shape (N, T, hidden_size), float32.
   * @param attention_mask  A tensor of shape (N, T) containing attention mask, int64.
   * @param cache_position  A tensor of shape (T,) containing cache positions, int64.
   * @param cache_kv  Fixed-size KV cache, vector of (key, value) pairs.
   * @return Return tuple (logits, kv_outputs...). Logits shape (N, T, vocab_size), float32.
   *         kv_outputs is a vector of (key_delta, value_delta) pairs for each layer.
   */
   std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
   ForwardLLM(Ort::Value inputs_embeds,
              Ort::Value attention_mask,
              const Ort::Value &cache_position,
              const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv);
  
  /** Create fixed-size KV cache for both legacy and KV-delta models.
   *
   * @param batch  Batch size (usually 1).
   * @param past_len  For legacy models: past sequence length (0 for first prefill).
   *                  For KV-delta models: ignored, uses max_total_len.
   * @return Return vector of (key, value) pairs with fixed cache dimensions.
   */
  std::vector<std::pair<Ort::Value, Ort::Value>>
  CreateEmptyKVCache(int64_t batch, int64_t past_len);

  /** Apply KV-delta in-place to fixed cache (for KV-delta models).
   *
   * @param cache_kv  Fixed-size KV cache to update, vector of (key, value) pairs.
   * @param kv_delta  KV deltas from current step, vector of (key_delta, value_delta) pairs.
   * @param cache_position  Cache position tensor indicating where to write deltas.
   */
  void ApplyKvDeltaInplace(std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
                          const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
                          const Ort::Value &cache_position);

  /** Check if using KV cache mode. Always returns true for FunASR-nano.
   */
  bool UseKVCache() const;

  /** Run the embedding model.
   *
   * @param input_ids  A tensor of shape (N, T) containing token IDs.
   * @return Return embeddings of shape (N, T, hidden_size)
   */
  Ort::Value ForwardEmbedding(Ort::Value input_ids);

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const;

  /** Return the hidden size of the model
   */
  int32_t HiddenSize() const;

  /** Return the maximum total sequence length (from metadata)
   */
  int32_t GetMaxTotalLen() const;

  /** It is lfr_window_size in metadata
   */
  int32_t LfrWindowSize() const;

  /** It is lfr_window_shift in metadata
   */
  int32_t LfrWindowShift() const;

  /** Return the maximum total sequence length
   */
  int64_t MaxTotalLen() const;

  // Unified LLM exported by Python may provide either cache_position or position_ids.
  // If neither exists in the model, HasPositionInput() is false and PositionInputRank() returns 0.
  // Rank is 1 (shape [S] / [1]) or 2 (shape [1,S] / [1,1]).
  bool HasPositionInput() const;
  int32_t PositionInputRank() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

  /** Check if embedding model is available
   */
  bool HasEmbeddingModel() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_H_

