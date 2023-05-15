// sherpa-onnx/csrc/offline-paraformer-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-paraformer-model.h"

#include <algorithm>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineParaformerModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init();
  }

  std::pair<Ort::Value, Ort::Value> Forward(Ort::Value features,
                                            Ort::Value features_length) {
    std::array<Ort::Value, 2> inputs = {std::move(features),
                                        std::move(features_length)};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return {std::move(out[0]), std::move(out[1])};
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t LfrWindowSize() const { return lfr_window_size_; }

  int32_t LfrWindowShift() const { return lfr_window_shift_; }

  const std::vector<float> &NegativeMean() const { return neg_mean_; }

  const std::vector<float> &InverseStdDev() const { return inv_stddev_; }

  OrtAllocator *Allocator() const { return allocator_; }

 private:
  void Init() {
    auto buf = ReadFile(config_.paraformer.model);

    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
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
    SHERPA_ONNX_READ_META_DATA(lfr_window_size_, "lfr_window_size");
    SHERPA_ONNX_READ_META_DATA(lfr_window_shift_, "lfr_window_shift");

    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(neg_mean_, "neg_mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(inv_stddev_, "inv_stddev");
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

  std::vector<float> neg_mean_;
  std::vector<float> inv_stddev_;

  int32_t vocab_size_ = 0;  // initialized in Init
  int32_t lfr_window_size_ = 0;
  int32_t lfr_window_shift_ = 0;
};

OfflineParaformerModel::OfflineParaformerModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineParaformerModel::~OfflineParaformerModel() = default;

std::pair<Ort::Value, Ort::Value> OfflineParaformerModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineParaformerModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OfflineParaformerModel::LfrWindowSize() const {
  return impl_->LfrWindowSize();
}
int32_t OfflineParaformerModel::LfrWindowShift() const {
  return impl_->LfrWindowShift();
}
const std::vector<float> &OfflineParaformerModel::NegativeMean() const {
  return impl_->NegativeMean();
}
const std::vector<float> &OfflineParaformerModel::InverseStdDev() const {
  return impl_->InverseStdDev();
}

OrtAllocator *OfflineParaformerModel::Allocator() const {
  return impl_->Allocator();
}

}  // namespace sherpa_onnx
