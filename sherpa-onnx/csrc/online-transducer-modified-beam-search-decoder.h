// sherpa-onnx/csrc/online-transducer-modified_beam-search-decoder.h
//
// Copyright (c)  2023  Pingfeng Luo
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/online-lm.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

class OnlineTransducerModifiedBeamSearchDecoder
    : public OnlineTransducerDecoder {
 public:
  OnlineTransducerModifiedBeamSearchDecoder(OnlineTransducerModel *model,
                                            OnlineLM *lm,
                                            int32_t max_active_paths,
                                            float lm_scale)
      : model_(model),
        lm_(lm),
        max_active_paths_(max_active_paths),
        lm_scale_(lm_scale) {}

  OnlineTransducerDecoderResult GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResult *r) const override;

  void Decode(Ort::Value encoder_out,
              std::vector<OnlineTransducerDecoderResult> *result) override;

  void Decode(Ort::Value encoder_out, OnlineStream **ss,
              std::vector<OnlineTransducerDecoderResult> *result) override;

  void UpdateDecoderOut(OnlineTransducerDecoderResult *result) override;

 private:
  OnlineTransducerModel *model_;  // Not owned
  OnlineLM *lm_;                  // Not owned

  int32_t max_active_paths_;
  float lm_scale_;  // used only when lm_ is not nullptr
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
