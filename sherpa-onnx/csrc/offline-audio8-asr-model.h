// sherpa-onnx/csrc/offline-audio8-asr-model.h
//
// Copyright (c)  2026  yuyun2000

#ifndef SHERPA_ONNX_CSRC_OFFLINE_AUDIO8_ASR_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_AUDIO8_ASR_MODEL_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-audio8-asr-model-config.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

// Audio8-ASR-0.1B: a Whisper-style audio encoder followed by a small Qwen2
// language model with a fixed-size KV cache.
class OfflineAudio8AsrModel {
 public:
  explicit OfflineAudio8AsrModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineAudio8AsrModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineAudio8AsrModel();

  /** Run the audio encoder.
   *
   * @param audios    Whisper log-mel, float32 [1, 128, 3000].
   * @param num_rows  int64 [1]. Number of encoder rows that correspond to the
   *                  real audio length; rows after it are padding.
   * @return float32 [1, 390, 1024].
   */
  Ort::Value ForwardEncoder(Ort::Value audios, Ort::Value num_rows) const;

  /** Run one prefill/decode step of the LM.
   *
   * @param input_ids       int64 [1, T].
   * @param audio_hidden    float32 [1, K, 1024], already pooled to K rows.
   * @param attention_mask  int64 [1, max_total_len].
   * @param cache_position  int64 [T].
   * @param cache_kv        (key, value) per layer, float32
   *                        [1, kv_heads, max_total_len, head_dim].
   * @return logits float32 [1, T, vocab_size] and the KV delta for this step.
   */
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLM(Ort::Value input_ids, Ort::Value audio_hidden,
            Ort::Value attention_mask, const Ort::Value &cache_position,
            const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv)
      const;

  /** Allocate a zeroed KV cache with max_total_len slots per layer. */
  std::vector<std::pair<Ort::Value, Ort::Value>> CreateEmptyKVCache(
      int64_t batch) const;

  /** Copy the KV delta of the current step into the cache, at
   * cache_position. */
  void ApplyKvDeltaInplace(
      std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
      const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
      const Ort::Value &cache_position) const;

  int32_t GetMaxTotalLen() const;
  int32_t NumLayers() const;
  int32_t NumKvHeads() const;
  int32_t HeadDim() const;
  int32_t VocabSize() const;

  OrtAllocator *Allocator() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_AUDIO8_ASR_MODEL_H_
