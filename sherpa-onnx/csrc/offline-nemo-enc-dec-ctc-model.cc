// sherpa-onnx/csrc/offline-nemo-enc-dec-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-nemo-enc-dec-ctc-model.h"

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class OfflineNemoEncDecCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) {
    SHERPA_ONNX_LOGE("here!\n");
    exit(-1);
  }

  std::pair<Ort::Value, Ort::Value> Forward(Ort::Value features,
                                            Ort::Value features_length) {
    Ort::Value a{nullptr};
    Ort::Value b{nullptr};
    return {std::move(a), std::move(b)};
  }

  int32_t VocabSize() const { return 0; }

  int32_t SubsamplingFactor() const { return 0; }

  OrtAllocator *Allocator() const { return nullptr; }

 private:
};

OfflineNemoEncDecCtcModel::OfflineNemoEncDecCtcModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineNemoEncDecCtcModel::~OfflineNemoEncDecCtcModel() = default;

std::pair<Ort::Value, Ort::Value> OfflineNemoEncDecCtcModel::Forward(
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

}  // namespace sherpa_onnx
