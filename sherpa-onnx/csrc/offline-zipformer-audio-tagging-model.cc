// sherpa-onnx/csrc/offline-zipformer-audio-tagging-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-zipformer-audio-tagging-model.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineZipformerAudioTaggingModel::Impl {
 public:
  explicit Impl(const AudioTaggingModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.zipformer.model);
    Init(buf.data(), buf.size());
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const AudioTaggingModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.zipformer.model);
    Init(buf.data(), buf.size());
  }
#endif

  Ort::Value Forward(Ort::Value features, Ort::Value features_length) {
    std::array<Ort::Value, 2> inputs = {std::move(features),
                                        std::move(features_length)};

    auto ans =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());
    return std::move(ans[0]);
  }

  int32_t NumEventClasses() const { return num_event_classes_; }

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
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    // get num_event_classes from the output[0].shape,
    // which is (N, num_event_classes)
    num_event_classes_ =
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];
  }

 private:
  AudioTaggingModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t num_event_classes_ = 0;
};

OfflineZipformerAudioTaggingModel::OfflineZipformerAudioTaggingModel(
    const AudioTaggingModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineZipformerAudioTaggingModel::OfflineZipformerAudioTaggingModel(
    AAssetManager *mgr, const AudioTaggingModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineZipformerAudioTaggingModel::~OfflineZipformerAudioTaggingModel() =
    default;

Ort::Value OfflineZipformerAudioTaggingModel::Forward(
    Ort::Value features, Ort::Value features_length) const {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineZipformerAudioTaggingModel::NumEventClasses() const {
  return impl_->NumEventClasses();
}

OrtAllocator *OfflineZipformerAudioTaggingModel::Allocator() const {
  return impl_->Allocator();
}

}  // namespace sherpa_onnx
