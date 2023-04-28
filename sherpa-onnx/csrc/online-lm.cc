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

void OnlineLM::ComputeLMScore(float scale, int32_t context_size,
                              std::vector<Hypotheses> *hyps) {
  Ort::AllocatorWithDefaultOptions allocator;

  for (auto &hyp : *hyps) {
    for (auto &h_m : hyp) {
      auto &h = h_m.second;
      auto &ys = h.ys;
      const int32_t token_len_to_score =
          h.ys.size() - context_size - h.cur_scored_pos;

      if (!h.lm_states_inited) {
        auto states = GetInitStates();
        h.lm_states.reserve(2);
        h.lm_states[0] = std::move(states[0]); // h
        h.lm_states[1] = std::move(states[2]); // c
        h.lm_states_inited = true;
      }

      if (token_len_to_score > 1) {
        std::array<int64_t, 2> x_shape{1, token_len_to_score};
        // shape of x and y are same
        Ort::Value x = Ort::Value::CreateTensor<int64_t>(
            allocator, x_shape.data(), x_shape.size());
        Ort::Value y = Ort::Value::CreateTensor<int64_t>(
            allocator, x_shape.data(), x_shape.size());

        int64_t *p_x = x.GetTensorMutableData<int64_t>();
        int64_t *p_y = y.GetTensorMutableData<int64_t>();
        std::copy(ys.begin() + context_size + h.cur_scored_pos, ys.end() - 1,
                  p_x);
        std::copy(ys.begin() + context_size + h.cur_scored_pos + 1, ys.end(),
                  p_y);

        std::vector<Ort::Value> states;
        states.push_back(std::move(h.lm_states[0].value));
        states.push_back(std::move(h.lm_states[1].value));

        // stream forward by RNN LM
        auto out = Rescore(std::move(x), std::move(y), std::move(states));

        // rescore hyp
        const float *p_nll = out.first.GetTensorData<float>();
        h.lm_log_prob = -scale * (*p_nll);

        // update RNN LM states
        h.lm_states.reserve(2);
        h.lm_states[0] = std::move(out.second[1]); // h
        h.lm_states[1] = std::move(out.second[2]); // c

        h.cur_scored_pos += token_len_to_score;
      }
    }
  }
}

}  // namespace sherpa_onnx
