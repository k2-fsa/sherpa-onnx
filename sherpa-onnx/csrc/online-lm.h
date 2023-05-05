// sherpa-onnx/csrc/online-lm.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_LM_H_
#define SHERPA_ONNX_CSRC_ONLINE_LM_H_

#include <memory>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/online-lm-config.h"

namespace sherpa_onnx {

class OnlineLM {
 public:
  virtual ~OnlineLM() = default;

  static std::unique_ptr<OnlineLM> Create(const OnlineLMConfig &config);

  virtual std::vector<Ort::Value> GetInitStates() = 0;

  /** Rescore a batch of sentences.
   *
   * @param x A 2-D tensor of shape (N, L) with data type int64.
   * @param y A 2-D tensor of shape (N, L) with data type int64.
   * @param states It contains the states for the LM model
   * @return Return a pair containingo
   *          - negative loglike
   *          - updated states
   *
   * Caution: It returns negative log likelihood (nll), not log likelihood
   */
  virtual std::pair<Ort::Value, std::vector<Ort::Value>> Rescore(
      Ort::Value x, Ort::Value y, std::vector<Ort::Value> states) = 0;

  // This function updates hyp.lm_lob_prob of hyps.
  //
  // @param scale LM score
  // @param context_size Context size of the transducer decoder model
  // @param hyps It is changed in-place.
  void ComputeLMScore(float scale, int32_t context_size,
                      std::vector<Hypotheses> *hyps);
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_LM_H_
