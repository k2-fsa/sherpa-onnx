// sherpa-onnx/csrc/online-lm.cc
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-lm.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/online-rnn-lm.h"

namespace sherpa_onnx {

std::unique_ptr<OnlineLM> OnlineLM::Create(const OnlineLMConfig &config) {
  return std::make_unique<OnlineRnnLM>(config);
}

void OnlineLM::ComputeLMScore(float scale, Hypothesis *hyp) {
  if (hyp->nn_lm_states.empty()) {
    auto init_states = GetInitStates();
    hyp->nn_lm_scores.value = std::move(init_states.first);
    hyp->nn_lm_states = Convert(std::move(init_states.second));
  }
  std::array<int64_t, 2> x_shape{1, 1};
  lm_x_.value = Ort::Value::CreateTensor<int64_t>(allocator_, x_shape.data(),
                                                  x_shape.size());

  *lm_x_.value.GetTensorMutableData<int64_t>() = hyp->ys.back();
  float *nn_lm_scores = hyp->nn_lm_scores.value.GetTensorMutableData<float>();
  hyp->lm_log_prob = nn_lm_scores[hyp->ys.back()];

  auto lm_out = ScoreToken(std::move(lm_x_.value), Convert(hyp->nn_lm_states));
  hyp->nn_lm_scores.value = std::move(lm_out.first);
  hyp->nn_lm_states = Convert(std::move(lm_out.second));
}

}  // namespace sherpa_onnx
