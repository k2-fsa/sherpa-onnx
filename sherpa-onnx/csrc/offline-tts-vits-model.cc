// sherpa-onnx/csrc/offline-tts-vits-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-vits-model.h"

#include <algorithm>
#include <string>
#include <utility>
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

  Ort::Value Run(Ort::Value x, int64_t sid, float speed) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> x_shape = x.GetTensorTypeAndShapeInfo().GetShape();
    if (x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Support only batch_size == 1. Given: %d",
                       static_cast<int32_t>(x_shape[0]));
      exit(-1);
    }

    int64_t len = x_shape[1];
    int64_t len_shape = 1;

    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &len, 1, &len_shape, 1);

    int64_t scale_shape = 1;
    float noise_scale = config_.vits.noise_scale;
    float length_scale = config_.vits.length_scale;
    float noise_scale_w = config_.vits.noise_scale_w;

    if (speed != 1 && speed > 0) {
      length_scale = 1. / speed;
    }

    Ort::Value noise_scale_tensor =
        Ort::Value::CreateTensor(memory_info, &noise_scale, 1, &scale_shape, 1);

    Ort::Value length_scale_tensor = Ort::Value::CreateTensor(
        memory_info, &length_scale, 1, &scale_shape, 1);

    Ort::Value noise_scale_w_tensor = Ort::Value::CreateTensor(
        memory_info, &noise_scale_w, 1, &scale_shape, 1);

    Ort::Value sid_tensor =
        Ort::Value::CreateTensor(memory_info, &sid, 1, &scale_shape, 1);

    std::vector<Ort::Value> inputs;
    inputs.reserve(6);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    inputs.push_back(std::move(noise_scale_tensor));
    inputs.push_back(std::move(length_scale_tensor));
    inputs.push_back(std::move(noise_scale_w_tensor));

    if (input_names_.size() == 6 && input_names_.back() == "sid") {
      inputs.push_back(std::move(sid_tensor));
    }

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return std::move(out[0]);
  }

  int32_t SampleRate() const { return sample_rate_; }

  bool AddBlank() const { return add_blank_; }

  std::string Punctuations() const { return punctuations_; }
  std::string Language() const { return language_; }
  int32_t NumSpeakers() const { return num_speakers_; }

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
    SHERPA_ONNX_READ_META_DATA(add_blank_, "add_blank");
    SHERPA_ONNX_READ_META_DATA(num_speakers_, "n_speakers");
    SHERPA_ONNX_READ_META_DATA_STR(punctuations_, "punctuation");
    SHERPA_ONNX_READ_META_DATA_STR(language_, "language");
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
  int32_t add_blank_;
  int32_t num_speakers_;
  std::string punctuations_;
  std::string language_;
};

OfflineTtsVitsModel::OfflineTtsVitsModel(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineTtsVitsModel::~OfflineTtsVitsModel() = default;

Ort::Value OfflineTtsVitsModel::Run(Ort::Value x, int64_t sid /*=0*/,
                                    float speed /*= 1.0*/) {
  return impl_->Run(std::move(x), sid, speed);
}

int32_t OfflineTtsVitsModel::SampleRate() const { return impl_->SampleRate(); }

bool OfflineTtsVitsModel::AddBlank() const { return impl_->AddBlank(); }

std::string OfflineTtsVitsModel::Punctuations() const {
  return impl_->Punctuations();
}

std::string OfflineTtsVitsModel::Language() const { return impl_->Language(); }

int32_t OfflineTtsVitsModel::NumSpeakers() const {
  return impl_->NumSpeakers();
}

}  // namespace sherpa_onnx
