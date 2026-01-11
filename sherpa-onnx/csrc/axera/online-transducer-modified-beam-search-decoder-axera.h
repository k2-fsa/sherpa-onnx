// sherpa-onnx/csrc/axera/online-transducer-modified-beam-search-decoder-axera.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXERA_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_AXERA_H_
#define SHERPA_ONNX_CSRC_AXERA_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_AXERA_H_

#include <vector>

#include "sherpa-onnx/csrc/axera/online-transducer-decoder-axera.h"
#include "sherpa-onnx/csrc/axera/online-zipformer-transducer-model-axera.h"

namespace sherpa_onnx {

class OnlineTransducerModifiedBeamSearchDecoderAxera
    : public OnlineTransducerDecoderAxera {
 public:
  explicit OnlineTransducerModifiedBeamSearchDecoderAxera(
      OnlineZipformerTransducerModelAxera *model, int32_t max_active_paths,
      int32_t unk_id = 2, float blank_penalty = 0.0)
      : model_(model),
        max_active_paths_(max_active_paths),
        unk_id_(unk_id),
        blank_penalty_(blank_penalty) {}

  OnlineTransducerDecoderResultAxera GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResultAxera *r) const override;

  void Decode(std::vector<float> encoder_out,
              OnlineTransducerDecoderResultAxera *result) const override;

 private:
  OnlineZipformerTransducerModelAxera *model_;  // Not owned
  int32_t max_active_paths_;
  int32_t unk_id_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_AXERA_H_
