// sherpa-onnx/csrc/offline-telespeech-ctc-model.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-telespeech-ctc-model.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineTeleSpeechCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.telespeech_ctc);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.telespeech_ctc);
    Init(buf.data(), buf.size());
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value /*features_length*/) {
    std::vector<int64_t> shape =
        features.GetTensorTypeAndShapeInfo().GetShape();

    if (static_cast<int32_t>(shape[0]) != 1) {
      SHERPA_ONNX_LOGE("This model supports only batch size 1. Given %d",
                       static_cast<int32_t>(shape[0]));
    }

    auto out = sess_->Run({}, input_names_ptr_.data(), &features, 1,
                          output_names_ptr_.data(), output_names_ptr_.size());

    std::vector<int64_t> logits_shape = {1};
    Ort::Value logits_length = Ort::Value::CreateTensor<int64_t>(
        allocator_, logits_shape.data(), logits_shape.size());

    int64_t *dst = logits_length.GetTensorMutableData<int64_t>();
    dst[0] = out[0].GetTensorTypeAndShapeInfo().GetShape()[0];

    // (T, B, C) -> (B, T, C)
    Ort::Value logits = Transpose01(allocator_, &out[0]);

    std::vector<Ort::Value> ans;
    ans.reserve(2);
    ans.push_back(std::move(logits));
    ans.push_back(std::move(logits_length));

    return ans;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

  OrtAllocator *Allocator() { return allocator_; }

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
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    {
      auto shape =
          sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      vocab_size_ = shape[2];
    }
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 4;
};

OfflineTeleSpeechCtcModel::OfflineTeleSpeechCtcModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTeleSpeechCtcModel::OfflineTeleSpeechCtcModel(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTeleSpeechCtcModel::~OfflineTeleSpeechCtcModel() = default;

std::vector<Ort::Value> OfflineTeleSpeechCtcModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineTeleSpeechCtcModel::VocabSize() const {
  return impl_->VocabSize();
}
int32_t OfflineTeleSpeechCtcModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

OrtAllocator *OfflineTeleSpeechCtcModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineTeleSpeechCtcModel::OfflineTeleSpeechCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineTeleSpeechCtcModel::OfflineTeleSpeechCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
