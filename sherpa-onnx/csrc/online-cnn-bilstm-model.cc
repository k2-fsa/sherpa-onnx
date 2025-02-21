// sherpa-onnx/csrc/online-cnn-bilstm-model.cc
//
// Copyright (c) 2024 Jian You (jianyou@cisco.com, Cisco Systems)

#include "sherpa-onnx/csrc/online-cnn-bilstm-model.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OnlineCNNBiLSTMModel::Impl {
 public:
  explicit Impl(const OnlinePunctuationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.cnn_bilstm);
    Init(buf.data(), buf.size());
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OnlinePunctuationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.cnn_bilstm);
    Init(buf.data(), buf.size());
  }
#endif

  std::pair<Ort::Value, Ort::Value> Forward(Ort::Value token_ids,
                                            Ort::Value valid_ids,
                                            Ort::Value label_lens) {
    std::array<Ort::Value, 3> inputs = {
        std::move(token_ids), std::move(valid_ids), std::move(label_lens)};

    auto ans =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());
    return {std::move(ans[0]), std::move(ans[1])};
  }

  OrtAllocator *Allocator() { return allocator_; }

  const OnlineCNNBiLSTMModelMetaData &GetModelMetadata() const {
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

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    SHERPA_ONNX_READ_META_DATA(meta_data_.comma_id, "COMMA");
    SHERPA_ONNX_READ_META_DATA(meta_data_.period_id, "PERIOD");
    SHERPA_ONNX_READ_META_DATA(meta_data_.quest_id, "QUESTION");

    // assert here, because we will use the constant value
    assert(meta_data_.comma_id == 1);
    assert(meta_data_.period_id == 2);
    assert(meta_data_.quest_id == 3);

    SHERPA_ONNX_READ_META_DATA(meta_data_.upper_id, "UPPER");
    SHERPA_ONNX_READ_META_DATA(meta_data_.cap_id, "CAP");
    SHERPA_ONNX_READ_META_DATA(meta_data_.mix_case_id, "MIX_CASE");

    assert(meta_data_.upper_id == 1);
    assert(meta_data_.cap_id == 2);
    assert(meta_data_.mix_case_id == 3);

    // output shape is (T', num_cases)
    meta_data_.num_cases =
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];
    meta_data_.num_punctuations =
        sess_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape()[1];
  }

 private:
  OnlinePunctuationModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  OnlineCNNBiLSTMModelMetaData meta_data_;
};

OnlineCNNBiLSTMModel::OnlineCNNBiLSTMModel(
    const OnlinePunctuationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OnlineCNNBiLSTMModel::OnlineCNNBiLSTMModel(
    AAssetManager *mgr, const OnlinePunctuationModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OnlineCNNBiLSTMModel::~OnlineCNNBiLSTMModel() = default;

std::pair<Ort::Value, Ort::Value> OnlineCNNBiLSTMModel::Forward(
    Ort::Value token_ids, Ort::Value valid_ids, Ort::Value label_lens) const {
  return impl_->Forward(std::move(token_ids), std::move(valid_ids),
                        std::move(label_lens));
}

OrtAllocator *OnlineCNNBiLSTMModel::Allocator() const {
  return impl_->Allocator();
}

const OnlineCNNBiLSTMModelMetaData &OnlineCNNBiLSTMModel::GetModelMetadata()
    const {
  return impl_->GetModelMetadata();
}

}  // namespace sherpa_onnx
