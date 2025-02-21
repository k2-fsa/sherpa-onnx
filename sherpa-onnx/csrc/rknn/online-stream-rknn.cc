// sherpa-onnx/csrc/rknn/online-stream-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/online-stream-rknn.h"

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

 private:
  rknn_context encoder_ctx_;
  rknn_context decoder_ctx_;
  rknn_context joiner_ctx_;
  std::vector<std::vector<uint8_t>> states_;
};

OnlineStreamRknn::OnlineStreamRknn(
    const FeatureExtractorConfig &config /*= {}*/,
    ContextGraphPtr context_graph /*= nullptr*/)
    : OnlineStream(config, context_graph) {}

OnlineStreamRknn::~OnlineStreamRknn() = default;

void OnlineStreamRknn::SetZipformerEncoderStates(
    std::vector<std::vector<uint8_t>> states) const {
  impl_->SetZipformerEncoderStates(std::move(states));
}

std::vector<std::vector<uint8_t>> &OnlineStreamRknn::GetZipformerEncoderStates()
    const {
  return impl_->GetZipformerEncoderStates();
}

}  // namespace sherpa_onnx
