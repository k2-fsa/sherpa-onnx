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
    hyp->nn_lm_states = Convert(GetInitStates());
  }
  std::array<int64_t, 2> x_shape{1, 1};
  lm_x_.value = Ort::Value::CreateTensor<int64_t>(allocator_, x_shape.data(),
                                                  x_shape.size());
  *lm_x_.value.GetTensorMutableData<int64_t>() =
      hyp->ys.back();  // get latest y in hyp
  auto lm_out =
      ScoreToken(std::move(lm_x_.value), std::move(Convert(hyp->nn_lm_states)));
  hyp->lm_log_prob += (*lm_out.first.GetTensorData<float>()) * scale;
  hyp->nn_lm_states = std::move(Convert(std::move(lm_out.second)));
}

}  // namespace sherpa_onnx
