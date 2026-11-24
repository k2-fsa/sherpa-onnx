// sherpa-onnx/csrc/offline-w2v-bert-ctc-model.h
//
// Copyright (c)  2026  graehl
#ifndef SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-ctc-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

/** A w2v-BERT 2.0 CTC model (transformers' Wav2Vec2BertForCTC) exported by
 * scripts/w2v-bert/export_onnx.py (metadata version 2 or later).
 *
 * The model takes "input_features", float32 (N, T, 160), and
 * "input_lengths", int64 (N,), and returns logits (N, T_out, vocab_size),
 * log-softmax normalized in the graph, and their lengths, int64 (N,). It masks
 * padding itself, so with fp32 weights (model.onnx) a padded batch gives each
 * utterance the logits it gets alone. model.int8.onnx quantizes activations
 * dynamically, with ranges taken over the whole batch, so its output depends on
 * the other utterances in the batch: batched transcripts can differ from single
 * ones by a word or a character. On 200 FLEURS Galician recordings, batches of
 * 8 and of 32 changed about a third of the transcripts but not the word error
 * rate. Decode one utterance per call where exact reproducibility matters.
 *
 * Its input is what SeamlessM4TFeatureExtractor computes: 80 Kaldi
 * filterbank energies per 10 ms from samples scaled to 16 bits (the
 * recognizer computes these), each bin normalized over the utterance
 * (NormalizeFeatures), and each two consecutive frames joined into one row
 * of 160 (by the recognizer, offline-recognizer-ctc-impl.h).
 */
class OfflineW2vBertCtcModel : public OfflineCtcModel {
 public:
  explicit OfflineW2vBertCtcModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineW2vBertCtcModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineW2vBertCtcModel() override;

  /** Run the forward method of the model.
   *
   * @param features  A tensor of shape (N, T, 160): frames normalized by
   *                  NormalizeFeatures(), each two consecutive ones joined
   *                  into one row. Padding may have any value.
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid rows in `features` before padding.
   *                         Its dtype is int64_t.
   *
   * @return Return a vector containing:
   *  - log_probs: A 3-D tensor of shape (N, T_out, vocab_size).
   *  - log_probs_length A 1-D tensor of shape (N,). Its dtype is int64_t
   */
  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) override;

  /** Return the vocabulary size of the model
   */
  int32_t VocabSize() const override;

  /** Filterbank frames (10 ms) per output frame, from the model's metadata.
   */
  int32_t SubsamplingFactor() const override;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const override;

  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_H_
