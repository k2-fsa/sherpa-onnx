// sherpa-onnx/csrc/on-rnn-lm.cc
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-rnn-lm.h"

#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OnlineRnnLM::Impl {
 public:
  explicit Impl(const OnlineLMConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_{},
        allocator_{} {
    Init(config);
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> Rescore(
      Ort::Value x, Ort::Value y, std::vector<Ort::Value> states) {
    std::array<Ort::Value, 4> inputs = {
        std::move(x), std::move(y), std::move(states[0]), std::move(states[1])};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    std::vector<Ort::Value> next_states;
    next_states.reserve(2);
    next_states.push_back(std::move(out[1]));
    next_states.push_back(std::move(out[2]));

    return {std::move(out[0]), std::move(next_states)};
  }

  std::vector<Ort::Value> GetInitStates() const {
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
    // shape of x and y are same
    Ort::Value x = Ort::Value::CreateTensor<int64_t>(allocator_, x_shape.data(),
                                                     x_shape.size());
    Ort::Value y = Ort::Value::CreateTensor<int64_t>(allocator_, x_shape.data(),
                                                     x_shape.size());
    *x.GetTensorMutableData<int64_t>() = sos_id_;
    *y.GetTensorMutableData<int64_t>() = sos_id_;

    std::vector<Ort::Value> states;
    states.push_back(std::move(h));
    states.push_back(std::move(c));
    auto pair = Rescore(std::move(x), std::move(y), std::move(states));

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

  std::vector<Ort::Value> init_states_;

  int32_t rnn_num_layers_ = 2;
  int32_t rnn_hidden_size_ = 512;
  int32_t sos_id_ = 1;
};

OnlineRnnLM::OnlineRnnLM(const OnlineLMConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OnlineRnnLM::~OnlineRnnLM() = default;

std::vector<Ort::Value> OnlineRnnLM::GetInitStates() {
  return impl_->GetInitStates();
}

std::pair<Ort::Value, std::vector<Ort::Value>> OnlineRnnLM::Rescore(
    Ort::Value x, Ort::Value y, std::vector<Ort::Value> states) {
  return impl_->Rescore(std::move(x), std::move(y), std::move(states));
}

}  // namespace sherpa_onnx
