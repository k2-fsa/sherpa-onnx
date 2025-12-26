// sherpa-onnx/csrc/axera/online-stream-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axcl/online-stream-axcl.h"

#include <utility>
#include <vector>

namespace sherpa_onnx {

class OnlineStreamAxcl::Impl {
 public:
  void SetZipformerEncoderStates(std::vector<std::vector<uint8_t>> states) {
    states_ = std::move(states);
  }

  std::vector<std::vector<uint8_t>> &GetZipformerEncoderStates() {
    return states_;
  }

  void SetZipformerResult(OnlineTransducerDecoderResultAxcl r) {
    result_ = std::move(r);
  }

  OnlineTransducerDecoderResultAxcl &GetZipformerResult() { return result_; }

 private:
  std::vector<std::vector<uint8_t>> states_;
  OnlineTransducerDecoderResultAxcl result_;
};

OnlineStreamAxcl::OnlineStreamAxcl(
    const FeatureExtractorConfig &config /*= {}*/,
    ContextGraphPtr context_graph /*= nullptr*/)
    : OnlineStream(config, context_graph), impl_(std::make_unique<Impl>()) {}

OnlineStreamAxcl::~OnlineStreamAxcl() = default;

void OnlineStreamAxcl::SetZipformerEncoderStates(
    std::vector<std::vector<uint8_t>> states) const {
  impl_->SetZipformerEncoderStates(std::move(states));
}

std::vector<std::vector<uint8_t>> &OnlineStreamAxcl::GetZipformerEncoderStates()
    const {
  return impl_->GetZipformerEncoderStates();
}

void OnlineStreamAxcl::SetZipformerResult(
    OnlineTransducerDecoderResultAxcl r) const {
  impl_->SetZipformerResult(std::move(r));
}

OnlineTransducerDecoderResultAxcl &OnlineStreamAxcl::GetZipformerResult()
    const {
  return impl_->GetZipformerResult();
}

}  // namespace sherpa_onnx
