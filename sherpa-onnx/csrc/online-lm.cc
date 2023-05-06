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

static std::vector<CopyableOrtValue> Convert(std::vector<Ort::Value> values) {
  std::vector<CopyableOrtValue> ans;
  ans.reserve(values.size());

  for (auto &v : values) {
    ans.emplace_back(std::move(v));
  }

  return ans;
}

static std::vector<Ort::Value> Convert(std::vector<CopyableOrtValue> values) {
  std::vector<Ort::Value> ans;
  ans.reserve(values.size());

  for (auto &v : values) {
    ans.emplace_back(std::move(v.value));
  }

  return ans;
}

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
      const int32_t token_num_in_chunk =
          ys.size() - context_size - h.cur_scored_pos - 1;

      if (token_num_in_chunk < 1) {
        continue;
      }

      if (h.nn_lm_states.empty()) {
        h.nn_lm_states = Convert(GetInitStates());
      }

      if (token_num_in_chunk >= h.lm_rescore_min_chunk) {
        std::array<int64_t, 2> x_shape{1, token_num_in_chunk};
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

        // streaming forward by NN LM
        auto out = Rescore(std::move(x), std::move(y),
                           Convert(std::move(h.nn_lm_states)));

        // update NN LM score in hyp
        const float *p_nll = out.first.GetTensorData<float>();
        h.lm_log_prob = -scale * (*p_nll);

        // update NN LM states in hyp
        h.nn_lm_states = Convert(std::move(out.second));

        h.cur_scored_pos += token_num_in_chunk;
      }
    }
  }
}

}  // namespace sherpa_onnx
