// sherpa-onnx/csrc/offline-zipformer-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-zipformer-ctc-model.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineZipformerCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.zipformer_ctc.model);
    Init(buf.data(), buf.size());
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.zipformer_ctc.model);
    Init(buf.data(), buf.size());
  }
#endif

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    std::array<Ort::Value, 2> inputs = {std::move(features),
                                        std::move(features_length)};

    return sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
  }

  int32_t VocabSize() const { return vocab_size_; }
  int32_t SubsamplingFactor() const { return 4; }

  OrtAllocator *Allocator() const { return allocator_; }

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

    // get vocab size from the output[0].shape, which is (N, T, vocab_size)
    vocab_size_ =
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2];
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
};

OfflineZipformerCtcModel::OfflineZipformerCtcModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineZipformerCtcModel::OfflineZipformerCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineZipformerCtcModel::~OfflineZipformerCtcModel() = default;

std::vector<Ort::Value> OfflineZipformerCtcModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineZipformerCtcModel::VocabSize() const {
  return impl_->VocabSize();
}

OrtAllocator *OfflineZipformerCtcModel::Allocator() const {
  return impl_->Allocator();
}

int32_t OfflineZipformerCtcModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

}  // namespace sherpa_onnx
