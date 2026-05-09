// sherpa-onnx/csrc/axera/online-transducer-greedy-search-decoder-axera.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXERA_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_AXERA_H_

#include <vector>

#include "sherpa-onnx/csrc/axera/online-transducer-decoder-axera.h"
#include "sherpa-onnx/csrc/axera/online-transducer-greedy-search-decoder-axera.h"
#include "sherpa-onnx/csrc/axera/online-zipformer-transducer-model-axera.h"

namespace sherpa_onnx {

class OnlineTransducerGreedySearchDecoderAxera
    : public OnlineTransducerDecoderAxera {
 public:
  explicit OnlineTransducerGreedySearchDecoderAxera(
      OnlineZipformerTransducerModelAxera *model, int32_t unk_id = 2,
      float blank_penalty = 0.0)
      : model_(model), unk_id_(unk_id), blank_penalty_(blank_penalty) {}

  OnlineTransducerDecoderResultAxera GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResultAxera *r) const override;

  void Decode(std::vector<float> encoder_out,
              OnlineTransducerDecoderResultAxera *result) const override;

 private:
  OnlineZipformerTransducerModelAxera *model_;  // Not owned
  int32_t unk_id_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_ONLINE_TRANSDUCER_GREEDY_SEARCH_DECODER_AXERA_H_
