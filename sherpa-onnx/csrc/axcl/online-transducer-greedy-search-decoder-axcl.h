// sherpa-onnx/csrc/axera/online-transducer-greedy-search-decoder-axera.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_AXCL_H_

#include <vector>

#include "sherpa-onnx/csrc/axcl/online-transducer-decoder-axcl.h"
#include "sherpa-onnx/csrc/axcl/online-transducer-greedy-search-decoder-axcl.h"
#include "sherpa-onnx/csrc/axcl/online-zipformer-transducer-model-axcl.h"

namespace sherpa_onnx {

class OnlineTransducerGreedySearchDecoderAxcl
    : public OnlineTransducerDecoderAxcl {
 public:
  explicit OnlineTransducerGreedySearchDecoderAxcl(
      OnlineZipformerTransducerModelAxcl *model, int32_t unk_id = 2,
      float blank_penalty = 0.0)
      : model_(model), unk_id_(unk_id), blank_penalty_(blank_penalty) {}

  OnlineTransducerDecoderResultAxcl GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResultAxcl *r) const override;

  void Decode(std::vector<float> encoder_out,
              OnlineTransducerDecoderResultAxcl *result) const override;

 private:
  OnlineZipformerTransducerModelAxcl *model_;  // Not owned
  int32_t unk_id_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_AXCL_H_
