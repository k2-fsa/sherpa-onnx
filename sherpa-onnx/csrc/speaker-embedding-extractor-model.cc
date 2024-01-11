// sherpa-onnx/csrc/speaker-embedding-extractor-model.cc
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/speaker-embedding-extractor-model.h"

#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-model-meta-data.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorModel::Impl {
 public:
  explicit Impl(const SpeakerEmbeddingExtractorConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.model);
      Init(buf.data(), buf.size());
    }
  }

  Ort::Value Compute(Ort::Value x) const {
    std::array<Ort::Value, 1> inputs = {std::move(x)};

    auto outputs =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());
    return std::move(outputs[0]);
  }

  const SpeakerEmbeddingExtractorModelMetaData &GetMetaData() const {
    return meta_data_;
  }

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
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.output_dim, "output_dim");
    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA(meta_data_.normalize_samples,
                               "normalize_samples");
    SHERPA_ONNX_READ_META_DATA_STR(meta_data_.language, "language");

    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(
        meta_data_.feature_normalize_type, "feature_normalize_type", "");

    std::string framework;
    SHERPA_ONNX_READ_META_DATA_STR(framework, "framework");
    if (framework != "wespeaker" && framework != "3d-speaker") {
      SHERPA_ONNX_LOGE("Expect a wespeaker or a 3d-speaker model, given: %s",
                       framework.c_str());
      exit(-1);
    }
  }

 private:
  SpeakerEmbeddingExtractorConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  SpeakerEmbeddingExtractorModelMetaData meta_data_;
};

SpeakerEmbeddingExtractorModel::SpeakerEmbeddingExtractorModel(
    const SpeakerEmbeddingExtractorConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

SpeakerEmbeddingExtractorModel::~SpeakerEmbeddingExtractorModel() = default;

const SpeakerEmbeddingExtractorModelMetaData &
SpeakerEmbeddingExtractorModel::GetMetaData() const {
  return impl_->GetMetaData();
}

Ort::Value SpeakerEmbeddingExtractorModel::Compute(Ort::Value x) const {
  return impl_->Compute(std::move(x));
}

}  // namespace sherpa_onnx
