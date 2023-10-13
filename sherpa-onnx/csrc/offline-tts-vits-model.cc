// sherpa-onnx/csrc/offline-tts-vits-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-vits-model.h"

#include <algorithm>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

class OfflineTtsVitsModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config.vits.model);
    Init(buf.data(), buf.size());
  }

  Ort::Value Run(Ort::Value x) { return Ort::Value(nullptr); }

  int32_t SampleRate() const { return sample_rate_; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---vits model---\n";
      PrintModelMetadata(os, meta_data);
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(sample_rate_, "sample_rate");
  }

 private:
  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t sample_rate_;
};

OfflineTtsVitsModel::OfflineTtsVitsModel(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineTtsVitsModel::~OfflineTtsVitsModel() = default;

Ort::Value OfflineTtsVitsModel::Run(Ort::Value x) {
  return impl_->Run(std::move(x));
}

int32_t OfflineTtsVitsModel::SampleRate() const { return impl_->SampleRate(); }

}  // namespace sherpa_onnx
