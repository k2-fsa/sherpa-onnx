// sherpa-onnx/csrc/silero-vad-model.h
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/silero-vad-model.h"

namespace sherpa_onnx {

class SileroVadModel::Impl {
 public:
  Impl(const VadModelConfig &config) : config_(config) {}

  void Reset() {}

  bool IsSpeech(const float *samples, int32_t n) { return true; }

 private:
  VadModelConfig config_;
};

SileroVadModel::SileroVadModel(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

SileroVadModel::~SileroVadModel() = default;

void SileroVadModel::Reset() { return impl_->Reset(); }

bool SileroVadModel::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

}  // namespace sherpa_onnx
