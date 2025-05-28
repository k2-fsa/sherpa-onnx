// sherpa-onnx/csrc/on-rnn-lm.cc
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-rnn-lm.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/lodr-fst.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OnlineRnnLM::Impl {
 public:
  explicit Impl(const OnlineLMConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_{GetSessionOptions(config)},
        allocator_{} {
    Init(config);
  }

  // shallow fusion scoring function
  void ComputeLMScoreSF(float scale, Hypothesis *hyp) {
    if (hyp->nn_lm_states.empty()) {
      auto init_states = GetInitStatesSF();
      hyp->nn_lm_scores.value = std::move(init_states.first);
      hyp->nn_lm_states = Convert(std::move(init_states.second));
      // if LODR enabled, we need to initialize the LODR state
      if (lodr_fst_ != nullptr) {
        hyp->lodr_state = std::make_unique<LodrStateCost>(lodr_fst_.get());
      }
    }

    // get lm score for cur token given the hyp->ys[:-1] and save to lm_log_prob
    const float *nn_lm_scores = hyp->nn_lm_scores.value.GetTensorData<float>();
    hyp->lm_log_prob += nn_lm_scores[hyp->ys.back()] * scale;

    // if LODR enabled, we need to update the LODR state
    if (lodr_fst_ != nullptr) {
      auto next_lodr_state = std::make_unique<LodrStateCost>(
                            hyp->lodr_state->ForwardOneStep(hyp->ys.back()));
      // calculate the score of the latest token
      auto score = next_lodr_state->Score() - hyp->lodr_state->Score();
      hyp->lodr_state = std::move(next_lodr_state);
      // apply LODR to hyp score
      hyp->lm_log_prob += score * config_.lodr_scale;
    }

    // get lm scores for next tokens given the hyp->ys[:] and save to
    // nn_lm_scores
    std::array<int64_t, 2> x_shape{1, 1};
    Ort::Value x = Ort::Value::CreateTensor<int64_t>(allocator_, x_shape.data(),
                                                     x_shape.size());
    *x.GetTensorMutableData<int64_t>() = hyp->ys.back();
    auto lm_out = ScoreToken(std::move(x), Convert(hyp->nn_lm_states));
    hyp->nn_lm_scores.value = std::move(lm_out.first);
    hyp->nn_lm_states = Convert(std::move(lm_out.second));
  }

  // classic rescore function
  void ComputeLMScore(float scale, int32_t context_size,
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

          Ort::Value x = Ort::Value::CreateTensor<int64_t>(
              allocator, x_shape.data(), x_shape.size());
          int64_t *p_x = x.GetTensorMutableData<int64_t>();
          std::copy(ys.begin() + context_size + h.cur_scored_pos, ys.end() - 1,
                    p_x);

          // streaming forward by NN LM
          auto out =
              ScoreToken(std::move(x), Convert(std::move(h.nn_lm_states)));

          // update NN LM score in hyp
          const float *p_nll = out.first.GetTensorData<float>();
          h.lm_log_prob = -scale * (*p_nll);

          // apply LODR to hyp score
          if (lodr_fst_ != nullptr) {
            // We scale LODR scale with LM scale to replicate Icefall code
            lodr_fst_->ComputeScore(config_.lodr_scale*scale, &h, context_size);
          }

          // update NN LM states in hyp
          h.nn_lm_states = Convert(std::move(out.second));

          h.cur_scored_pos += token_num_in_chunk;
        }
      }
    }
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> ScoreToken(
      Ort::Value x, std::vector<Ort::Value> states) {
    std::array<Ort::Value, 3> inputs = {std::move(x), std::move(states[0]),
                                        std::move(states[1])};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    std::vector<Ort::Value> next_states;
    next_states.reserve(2);
    next_states.push_back(std::move(out[1]));
    next_states.push_back(std::move(out[2]));

    return {std::move(out[0]), std::move(next_states)};
  }

  // get init states for shallow fusion
  std::pair<Ort::Value, std::vector<Ort::Value>> GetInitStatesSF() {
    std::vector<Ort::Value> ans;
    ans.reserve(init_states_.size());
    for (auto &s : init_states_) {
      ans.emplace_back(View(&s));
    }
    return {View(&init_scores_.value), std::move(ans)};
  }

  // get init states for classic rescore
  std::vector<Ort::Value> GetInitStates() {
    std::vector<Ort::Value> ans;
    ans.reserve(init_states_.size());

    for (const auto &s : init_states_) {
      ans.emplace_back(Clone(allocator_, &s));
    }

    return ans;
  }

 private:
  void Init(const OnlineLMConfig &config) {
    auto buf = ReadFile(config_.model);

    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(rnn_num_layers_, "num_layers");
    SHERPA_ONNX_READ_META_DATA(rnn_hidden_size_, "hidden_size");
    SHERPA_ONNX_READ_META_DATA(sos_id_, "sos_id");

    ComputeInitStates();

    if (!config_.lodr_fst.empty()) {
      lodr_fst_ = std::make_unique<LodrFst>(LodrFst(config_.lodr_fst,
                                                    config_.lodr_backoff_id));
    }
  }

  void ComputeInitStates() {
    constexpr int32_t kBatchSize = 1;
    std::array<int64_t, 3> h_shape{rnn_num_layers_, kBatchSize,
                                   rnn_hidden_size_};
    std::array<int64_t, 3> c_shape{rnn_num_layers_, kBatchSize,
                                   rnn_hidden_size_};
    Ort::Value h = Ort::Value::CreateTensor<float>(allocator_, h_shape.data(),
                                                   h_shape.size());
    Ort::Value c = Ort::Value::CreateTensor<float>(allocator_, c_shape.data(),
                                                   c_shape.size());
    Fill<float>(&h, 0);
    Fill<float>(&c, 0);
    std::array<int64_t, 2> x_shape{1, 1};
    Ort::Value x = Ort::Value::CreateTensor<int64_t>(allocator_, x_shape.data(),
                                                     x_shape.size());
    *x.GetTensorMutableData<int64_t>() = sos_id_;

    std::vector<Ort::Value> states;
    states.push_back(std::move(h));
    states.push_back(std::move(c));
    auto pair = ScoreToken(std::move(x), std::move(states));

    init_scores_.value = std::move(pair.first);  // only used during
                                                 // shallow fusion
    init_states_ = std::move(pair.second);
  }

 private:
  OnlineLMConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  CopyableOrtValue init_scores_;
  std::vector<Ort::Value> init_states_;

  int32_t rnn_num_layers_ = 2;
  int32_t rnn_hidden_size_ = 512;
  int32_t sos_id_ = 1;

  std::unique_ptr<LodrFst> lodr_fst_;
};

OnlineRnnLM::OnlineRnnLM(const OnlineLMConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OnlineRnnLM::~OnlineRnnLM() = default;

// classic rescore state init
std::vector<Ort::Value> OnlineRnnLM::GetInitStates() {
  return impl_->GetInitStates();
}

// shallow fusion state init
std::pair<Ort::Value, std::vector<Ort::Value>> OnlineRnnLM::GetInitStatesSF() {
  return impl_->GetInitStatesSF();
}

std::pair<Ort::Value, std::vector<Ort::Value>> OnlineRnnLM::ScoreToken(
    Ort::Value x, std::vector<Ort::Value> states) {
  return impl_->ScoreToken(std::move(x), std::move(states));
}

// classic rescore scores
void OnlineRnnLM::ComputeLMScore(float scale, int32_t context_size,
                                 std::vector<Hypotheses> *hyps) {
  return impl_->ComputeLMScore(scale, context_size, hyps);
}

// shallow fusion scores
void OnlineRnnLM::ComputeLMScoreSF(float scale, Hypothesis *hyp) {
  return impl_->ComputeLMScoreSF(scale, hyp);
}

}  // namespace sherpa_onnx
