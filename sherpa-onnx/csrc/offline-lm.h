// sherpa-onnx/csrc/offline-lm.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_LM_H_
#define SHERPA_ONNX_CSRC_OFFLINE_LM_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/lodr-fst.h"
#include "sherpa-onnx/csrc/offline-lm-config.h"

namespace sherpa_onnx {

class OfflineLM {
 public:
  explicit OfflineLM(const OfflineLMConfig &config) : config_(config) {
    if (!config_.lodr_fst.empty()) {
      try {
        lodr_fst_ = std::make_unique<LodrFst>(LodrFst(config_.lodr_fst,
                                                    config_.lodr_backoff_id));
      } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load LODR FST from: " +
                                  config_.lodr_fst + ". Error: " + e.what());
      }
    }
  }
  virtual ~OfflineLM() = default;

  static std::unique_ptr<OfflineLM> Create(const OfflineLMConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineLM> Create(Manager *mgr,
                                           const OfflineLMConfig &config);

  /** Rescore a batch of sentences.
   *
   * @param x A 2-D tensor of shape (N, L) with data type int64.
   * @param x_lens A 1-D tensor of shape (N,) with data type int64.
   *               It contains number of valid tokens in x before padding.
   * @return Return a 1-D tensor of shape (N,) containing the negative log
   *         likelihood of each utterance. Its data type is float32.
   *
   * Caution: It returns negative log likelihood (nll), not log likelihood
   */
  virtual Ort::Value Rescore(Ort::Value x, Ort::Value x_lens) = 0;

  // This function updates hyp.lm_lob_prob of hyps.
  //
  // @param scale LM score
  // @param context_size Context size of the transducer decoder model
  // @param hyps It is changed in-place.
  void ComputeLMScore(float scale, int32_t context_size,
                      std::vector<Hypotheses> *hyps);

 private:
  std::unique_ptr<LodrFst> lodr_fst_;
  float lodr_scale_;
  OfflineLMConfig config_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_LM_H_
