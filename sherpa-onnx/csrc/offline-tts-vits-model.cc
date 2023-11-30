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

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.vits.model);
    Init(buf.data(), buf.size());
  }
#endif

  Ort::Value Run(Ort::Value x, int64_t sid, float speed) {
    if (is_piper_) {
      return RunVitsPiper(std::move(x), sid, speed);
    }

    return RunVits(std::move(x), sid, speed);
  }

  int32_t SampleRate() const { return sample_rate_; }

  bool AddBlank() const { return add_blank_; }

  std::string Punctuations() const { return punctuations_; }
  std::string Language() const { return language_; }
  std::string Voice() const { return voice_; }
  bool IsPiper() const { return is_piper_; }
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
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(add_blank_, "add_blank", 0);
    SHERPA_ONNX_READ_META_DATA(num_speakers_, "n_speakers");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(punctuations_, "punctuation",
                                                "");
    SHERPA_ONNX_READ_META_DATA_STR(language_, "language");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(voice_, "voice", "");

    std::string comment;
    SHERPA_ONNX_READ_META_DATA_STR(comment, "comment");
    if (comment.find("piper") != std::string::npos ||
        comment.find("coqui") != std::string::npos) {
      is_piper_ = true;
    }
  }

  Ort::Value RunVitsPiper(Ort::Value x, int64_t sid, float speed) {
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

    float noise_scale = config_.vits.noise_scale;
    float length_scale = config_.vits.length_scale;
    float noise_scale_w = config_.vits.noise_scale_w;

    if (speed != 1 && speed > 0) {
      length_scale = 1. / speed;
    }
    std::array<float, 3> scales = {noise_scale, length_scale, noise_scale_w};

    int64_t scale_shape = 3;

    Ort::Value scales_tensor = Ort::Value::CreateTensor(
        memory_info, scales.data(), scales.size(), &scale_shape, 1);

    int64_t sid_shape = 1;
    Ort::Value sid_tensor =
        Ort::Value::CreateTensor(memory_info, &sid, 1, &sid_shape, 1);

    std::vector<Ort::Value> inputs;
    inputs.reserve(4);
    inputs.push_back(std::move(x));
    inputs.push_back(std::move(x_length));
    inputs.push_back(std::move(scales_tensor));

    if (input_names_.size() == 4 && input_names_.back() == "sid") {
      inputs.push_back(std::move(sid_tensor));
    }

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return std::move(out[0]);
  }

  Ort::Value RunVits(Ort::Value x, int64_t sid, float speed) {
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
  std::string voice_;

  bool is_piper_ = false;
};

OfflineTtsVitsModel::OfflineTtsVitsModel(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineTtsVitsModel::OfflineTtsVitsModel(AAssetManager *mgr,
                                         const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

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
std::string OfflineTtsVitsModel::Voice() const { return impl_->Voice(); }

bool OfflineTtsVitsModel::IsPiper() const { return impl_->IsPiper(); }

int32_t OfflineTtsVitsModel::NumSpeakers() const {
  return impl_->NumSpeakers();
}

}  // namespace sherpa_onnx
