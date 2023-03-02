// sherpa-onnx/csrc/online-transducer-modified_beam-search-decoder.h
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

class OnlineTransducerModifiedBeamSearchDecoder
    : public OnlineTransducerDecoder {
 public:
  OnlineTransducerModifiedBeamSearchDecoder(OnlineTransducerModel *model,
                                            int32_t max_active_paths)
      : model_(model), max_active_paths_(max_active_paths) {}

  OnlineTransducerDecoderResult GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResult *r) const override;

  void Decode(Ort::Value encoder_out,
              std::vector<OnlineTransducerDecoderResult> *result) override;

  void UpdateDecoderOut(OnlineTransducerDecoderResult *result) override;

 private:
  OnlineTransducerModel *model_;  // Not owned
  int32_t max_active_paths_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
