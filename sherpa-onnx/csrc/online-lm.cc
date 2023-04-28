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

  for (auto &h : *hyps) {
    for (auto &tok : h) {
      auto &t = tok.second;
      auto &ys = t.ys;
      const int32_t token_len_to_score =
          t.ys.size() - context_size - t.cur_scored_pos;

      if (!t.lm_states_inited) {
        auto states = GetInitStates();
        std::vector<CopyableOrtValue> updated_states;
        updated_states.push_back(std::move(states[0]));
        updated_states.push_back(std::move(states[1]));
        t.lm_states = updated_states;
        t.lm_states_inited = true;
      }

      if (token_len_to_score > 0) {
        std::array<int64_t, 2> x_shape{1, token_len_to_score};
        // shape of x and y are same
        Ort::Value x = Ort::Value::CreateTensor<int64_t>(
            allocator, x_shape.data(), x_shape.size());
        Ort::Value y = Ort::Value::CreateTensor<int64_t>(
            allocator, x_shape.data(), x_shape.size());

        int64_t *p_x = x.GetTensorMutableData<int64_t>();
        int64_t *p_y = y.GetTensorMutableData<int64_t>();
        std::fill(p_x, p_x + token_len_to_score, 0);
        std::fill(p_y, p_y + token_len_to_score, 0);

        std::copy(ys.begin() + context_size + t.cur_scored_pos, ys.end() - 1,
                  p_x);
        std::copy(ys.begin() + context_size + t.cur_scored_pos + 1, ys.end(),
                  p_y);

        std::vector<Ort::Value> states;
        states.push_back(std::move(t.lm_states[0].value));
        states.push_back(std::move(t.lm_states[1].value));

        // stream forward by RNN LM
        auto out = Rescore(std::move(x), std::move(y), std::move(states));

        // rescore hyp
        const float *p_nll = out.first.GetTensorData<float>();
        t.lm_log_prob = -scale * (*p_nll);

        // update RNN LM states
        std::vector<CopyableOrtValue> updated_states;
        updated_states.push_back(std::move(out.second[1]));
        updated_states.push_back(std::move(out.second[2]));
        t.lm_states = updated_states;

        t.cur_scored_pos += token_len_to_score;
      }
    }
  }
}

}  // namespace sherpa_onnx
