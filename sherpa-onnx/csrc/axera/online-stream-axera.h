// sherpa-onnx/csrc/axera/online-stream-axera.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AXERA_ONLINE_STREAM_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_ONLINE_STREAM_AXERA_H_
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/axera/online-transducer-decoder-axera.h"

namespace sherpa_onnx {

class OnlineStreamAxera : public OnlineStream {
 public:
  explicit OnlineStreamAxera(const FeatureExtractorConfig &config = {},
                            ContextGraphPtr context_graph = nullptr);

  ~OnlineStreamAxera();

  void SetZipformerEncoderStates(
      std::vector<std::vector<uint8_t>> states) const;

  std::vector<std::vector<uint8_t>> &GetZipformerEncoderStates() const;

  void SetZipformerResult(OnlineTransducerDecoderResultAxera r) const;

  OnlineTransducerDecoderResultAxera &GetZipformerResult() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_ONLINE_STREAM_AXERA_H_
