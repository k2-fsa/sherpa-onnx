#ifndef SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_H_

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

  /** Run the LLM model (legacy mode, single model).
   *
   * @param inputs_embeds  A tensor of shape (N, T, hidden_size).
   * @param attention_mask  A tensor of shape (N, T) containing attention mask.
   * @return Return logits of shape (N, T, vocab_size)
   */
  Ort::Value ForwardLLM(Ort::Value inputs_embeds, Ort::Value attention_mask);

  /** Run the LLM prefill model (KV cache mode).
   *
   * @param inputs_embeds  A tensor of shape (N, T, hidden_size).
   * @param attention_mask  A tensor of shape (N, T) containing attention mask.
   * @return Return tuple (logits, past_key_values...). Logits shape (N, T, vocab_size).
   *         past_key_values is a vector of (key, value) pairs for each layer.
   */
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLMPrefill(Ort::Value inputs_embeds, Ort::Value attention_mask);

  /** Run the LLM decode model (KV cache mode).
   *
   * @param inputs_embeds  A tensor of shape (N, 1, hidden_size) for the next token.
   * @param attention_mask  A tensor of shape (N, total_seq_len) containing attention mask.
   * @param past_key_values  KV cache from previous steps, vector of (key, value) pairs.
   * @return Return tuple (logits, updated_past_key_values...). Logits shape (N, 1, vocab_size).
   */
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLMDecode(Ort::Value inputs_embeds, Ort::Value attention_mask,
                   const std::vector<std::pair<Ort::Value, Ort::Value>> &past_key_values);

  /** Check if using KV cache mode (prefill + decode) or legacy mode (single model)
   */
  bool UseKVCache() const;

  /** Run the embedding model (optional).
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

  /** It is lfr_window_size in metadata
   */
  int32_t LfrWindowSize() const;

  /** It is lfr_window_shift in metadata
   */
  int32_t LfrWindowShift() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

  /** Check if embedding model is available
   */
  bool HasEmbeddingModel() const;

  /** Get expected input type for prefill model
   */
  ONNXTensorElementDataType GetPrefillInputType() const;

  /** Get expected input type for decode model
   */
  ONNXTensorElementDataType GetDecodeInputType() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FUNASR_NANO_MODEL_H_

