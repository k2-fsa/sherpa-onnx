// sherpa-onnx/csrc/offline-rnn-lm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-rnn-lm.h"

#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

class OfflineRnnLM::Impl {
 public:
  explicit Impl(const OfflineRecognizerConfig &config)
      : config_(config.lm_config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_{GetSessionOptions(config.model_config)},
        allocator_{} {
    Init(config.lm_config);
  }

  Ort::Value Rescore(Ort::Value x, Ort::Value x_lens) {
    std::array<Ort::Value, 2> inputs = {std::move(x), std::move(x_lens)};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return std::move(out[0]);
  }

 private:
  void Init(const OfflineLMConfig &config) {
    auto buf = ReadFile(config_.model);

    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
  }

 private:
  OfflineLMConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineRnnLM::OfflineRnnLM(const OfflineRecognizerConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineRnnLM::~OfflineRnnLM() = default;

Ort::Value OfflineRnnLM::Rescore(Ort::Value x, Ort::Value x_lens) {
  return impl_->Rescore(std::move(x), std::move(x_lens));
}

}  // namespace sherpa_onnx
