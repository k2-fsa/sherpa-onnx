// sherpa-onnx/csrc/axera/online-stream-axera.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AXCL_ONLINE_STREAM_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_ONLINE_STREAM_AXCL_H_
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/axcl/online-transducer-decoder-axcl.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

class OnlineStreamAxcl : public OnlineStream {
 public:
  explicit OnlineStreamAxcl(const FeatureExtractorConfig &config = {},
                            ContextGraphPtr context_graph = nullptr);

  ~OnlineStreamAxcl();

  void SetZipformerEncoderStates(
      std::vector<std::vector<uint8_t>> states) const;

  std::vector<std::vector<uint8_t>> &GetZipformerEncoderStates() const;

  void SetZipformerResult(OnlineTransducerDecoderResultAxcl r) const;

  OnlineTransducerDecoderResultAxcl &GetZipformerResult() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_ONLINE_STREAM_AXCL_H_
