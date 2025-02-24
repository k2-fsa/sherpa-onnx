// sherpa-onnx/csrc/rknn/online-stream-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/online-stream-rknn.h"

#include <utility>
#include <vector>

namespace sherpa_onnx {

class OnlineStreamRknn::Impl {
 public:
  void SetZipformerEncoderStates(std::vector<std::vector<uint8_t>> states) {
    states_ = std::move(states);
  }

  std::vector<std::vector<uint8_t>> &GetZipformerEncoderStates() {
    return states_;
  }

  void SetZipformerResult(OnlineTransducerDecoderResultRknn r) {
    result_ = std::move(r);
  }

  OnlineTransducerDecoderResultRknn &GetZipformerResult() { return result_; }

 private:
  std::vector<std::vector<uint8_t>> states_;
  OnlineTransducerDecoderResultRknn result_;
};

OnlineStreamRknn::OnlineStreamRknn(
    const FeatureExtractorConfig &config /*= {}*/,
    ContextGraphPtr context_graph /*= nullptr*/)
    : OnlineStream(config, context_graph), impl_(std::make_unique<Impl>()) {}

OnlineStreamRknn::~OnlineStreamRknn() = default;

void OnlineStreamRknn::SetZipformerEncoderStates(
    std::vector<std::vector<uint8_t>> states) const {
  impl_->SetZipformerEncoderStates(std::move(states));
}

std::vector<std::vector<uint8_t>> &OnlineStreamRknn::GetZipformerEncoderStates()
    const {
  return impl_->GetZipformerEncoderStates();
}

void OnlineStreamRknn::SetZipformerResult(
    OnlineTransducerDecoderResultRknn r) const {
  impl_->SetZipformerResult(std::move(r));
}

OnlineTransducerDecoderResultRknn &OnlineStreamRknn::GetZipformerResult()
    const {
  return impl_->GetZipformerResult();
}

}  // namespace sherpa_onnx
