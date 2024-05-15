// sherpa-onnx/csrc/offline-nemo-enc-dec-ctc-model.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-nemo-enc-dec-ctc-model.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineNemoEncDecCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.nemo_ctc.model);
    Init(buf.data(), buf.size());
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.nemo_ctc.model);
    Init(buf.data(), buf.size());
  }
#endif

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    std::vector<int64_t> shape =
        features_length.GetTensorTypeAndShapeInfo().GetShape();

    Ort::Value out_features_length = Ort::Value::CreateTensor<int64_t>(
        allocator_, shape.data(), shape.size());

    const int64_t *src = features_length.GetTensorData<int64_t>();
    int64_t *dst = out_features_length.GetTensorMutableData<int64_t>();
    for (int64_t i = 0; i != shape[0]; ++i) {
      dst[i] = src[i] / subsampling_factor_;
    }

    // (B, T, C) -> (B, C, T)
    features = Transpose12(allocator_, &features);

    std::array<Ort::Value, 2> inputs = {std::move(features),
                                        std::move(features_length)};
    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    std::vector<Ort::Value> ans;
    ans.reserve(2);
    ans.push_back(std::move(out[0]));
    ans.push_back(std::move(out_features_length));
    return ans;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }

  OrtAllocator *Allocator() const { return allocator_; }

  std::string FeatureNormalizationMethod() const { return normalize_type_; }

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

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA_STR(normalize_type_, "normalize_type");
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
  int32_t subsampling_factor_ = 0;
  std::string normalize_type_;
};

OfflineNemoEncDecCtcModel::OfflineNemoEncDecCtcModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineNemoEncDecCtcModel::OfflineNemoEncDecCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineNemoEncDecCtcModel::~OfflineNemoEncDecCtcModel() = default;

std::vector<Ort::Value> OfflineNemoEncDecCtcModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineNemoEncDecCtcModel::VocabSize() const {
  return impl_->VocabSize();
}
int32_t OfflineNemoEncDecCtcModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

OrtAllocator *OfflineNemoEncDecCtcModel::Allocator() const {
  return impl_->Allocator();
}

std::string OfflineNemoEncDecCtcModel::FeatureNormalizationMethod() const {
  return impl_->FeatureNormalizationMethod();
}

}  // namespace sherpa_onnx
