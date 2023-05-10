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
  OnlineLM() : allocator_{} {}

  virtual ~OnlineLM() = default;

  static std::unique_ptr<OnlineLM> Create(const OnlineLMConfig &config);

  virtual std::pair<Ort::Value, std::vector<Ort::Value>> GetInitStates() = 0;

  /** ScoreToken a batch of sentences.
   *
   * @param x A 2-D tensor of shape (N, L) with data type int64.
   * @param lens A 2-D tensor of shape (N, L) with data type int64.
   * @param states It contains the states for the LM model
   * @return Return a pair containingo
   *          - log_prob of NN LM
   *          - updated states
   *
   */
  virtual std::pair<Ort::Value, std::vector<Ort::Value>> ScoreToken(
      Ort::Value x, std::vector<Ort::Value> states) = 0;

  /** This function updates lm_lob_prob and nn_lm_scores of hyp
   *
   * @param scale LM score
   * @param hyps It is changed in-place.
   *
   */
  void ComputeLMScore(float scale, Hypothesis *hyp);

  CopyableOrtValue lm_x_;
  Ort::AllocatorWithDefaultOptions allocator_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_LM_H_
