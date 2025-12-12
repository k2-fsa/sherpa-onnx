// sherpa-onnx/csrc/axera/online-stream-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/online-stream-axera.h"

#include <utility>
#include <vector>

namespace sherpa_onnx {

class OnlineStreamAxera::Impl {
 public:
  void SetZipformerEncoderStates(std::vector<std::vector<uint8_t>> states) {
    states_ = std::move(states);
  }

  std::vector<std::vector<uint8_t>> &GetZipformerEncoderStates() {
    return states_;
  }

  void SetZipformerResult(OnlineTransducerDecoderResultAxera r) {
    result_ = std::move(r);
  }

  OnlineTransducerDecoderResultAxera &GetZipformerResult() { return result_; }

 private:
  std::vector<std::vector<uint8_t>> states_;
  OnlineTransducerDecoderResultAxera result_;
};

OnlineStreamAxera::OnlineStreamAxera(
    const FeatureExtractorConfig &config /*= {}*/,
    ContextGraphPtr context_graph /*= nullptr*/)
    : OnlineStream(config, context_graph), impl_(std::make_unique<Impl>()) {}

OnlineStreamAxera::~OnlineStreamAxera() = default;

void OnlineStreamAxera::SetZipformerEncoderStates(
    std::vector<std::vector<uint8_t>> states) const {
  impl_->SetZipformerEncoderStates(std::move(states));
}

std::vector<std::vector<uint8_t>> &
OnlineStreamAxera::GetZipformerEncoderStates() const {
  return impl_->GetZipformerEncoderStates();
}

void OnlineStreamAxera::SetZipformerResult(
    OnlineTransducerDecoderResultAxera r) const {
  impl_->SetZipformerResult(std::move(r));
}

OnlineTransducerDecoderResultAxera &OnlineStreamAxera::GetZipformerResult()
    const {
  return impl_->GetZipformerResult();
}

}  // namespace sherpa_onnx
