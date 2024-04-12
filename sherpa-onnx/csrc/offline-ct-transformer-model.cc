// sherpa-onnx/csrc/offline-ct-transformer-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ct-transformer-model.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineCtTransformerModel::Impl {
 public:
  explicit Impl(const OfflinePunctuationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.ct_transformer);
    Init(buf.data(), buf.size());
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OfflinePunctuationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.ct_transformer);
    Init(buf.data(), buf.size());
  }
#endif

  Ort::Value Forward(Ort::Value text) {}

  OrtAllocator *Allocator() const { return allocator_; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
  }

 private:
  OfflinePunctuationModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineCtTransformerModel::OfflineCtTransformerModel(
    const OfflinePunctuationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineCtTransformerModel::OfflineCtTransformerModel(
    AAssetManager *mgr, const OfflinePunctuationModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineCtTransformerModel::~OfflineCtTransformerModel() = default;

Ort::Value OfflineCtTransformerModel::Forward(Ort::Value text) const {
  return impl_->Forward(std::move(text));
}

OrtAllocator *OfflineCtTransformerModel::Allocator() const {
  return impl_->Allocator();
}

}  // namespace sherpa_onnx
