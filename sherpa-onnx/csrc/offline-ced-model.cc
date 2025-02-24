// sherpa-onnx/csrc/offline-ced-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ced-model.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineCEDModel::Impl {
 public:
  explicit Impl(const AudioTaggingModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.ced);
    Init(buf.data(), buf.size());
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const AudioTaggingModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.ced);
    Init(buf.data(), buf.size());
  }
#endif

  Ort::Value Forward(Ort::Value features) {
    features = Transpose12(allocator_, &features);

    auto ans = sess_->Run({}, input_names_ptr_.data(), &features, 1,
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

OfflineCEDModel::OfflineCEDModel(const AudioTaggingModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineCEDModel::OfflineCEDModel(AAssetManager *mgr,
                                 const AudioTaggingModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineCEDModel::~OfflineCEDModel() = default;

Ort::Value OfflineCEDModel::Forward(Ort::Value features) const {
  return impl_->Forward(std::move(features));
}

int32_t OfflineCEDModel::NumEventClasses() const {
  return impl_->NumEventClasses();
}

OrtAllocator *OfflineCEDModel::Allocator() const { return impl_->Allocator(); }

}  // namespace sherpa_onnx
