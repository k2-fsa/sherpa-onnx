// sherpa-onnx/csrc/rknn/online-stream-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_RKNN_ONLINE_STREAM_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_ONLINE_STREAM_RKNN_H_
#include <memory>
#include <vector>

#include "rknn_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/rknn/online-transducer-decoder-rknn.h"

namespace sherpa_onnx {

class OnlineStreamRknn : public OnlineStream {
 public:
  explicit OnlineStreamRknn(const FeatureExtractorConfig &config = {},
                            ContextGraphPtr context_graph = nullptr);

  ~OnlineStreamRknn();

  void SetZipformerEncoderStates(
      std::vector<std::vector<uint8_t>> states) const;

  std::vector<std::vector<uint8_t>> &GetZipformerEncoderStates() const;

  void SetZipformerResult(OnlineTransducerDecoderResultRknn r) const;

  OnlineTransducerDecoderResultRknn &GetZipformerResult() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_ONLINE_STREAM_RKNN_H_
