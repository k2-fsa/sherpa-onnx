// sherpa-onnx/csrc/axera/online-transducer-modified-beam-search-decoder-axera.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXCL_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_AXCL_H_
#define SHERPA_ONNX_CSRC_AXCL_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_AXCL_H_

#include <vector>

#include "sherpa-onnx/csrc/axcl/online-transducer-decoder-axcl.h"
#include "sherpa-onnx/csrc/axcl/online-zipformer-transducer-model-axcl.h"

namespace sherpa_onnx {

class OnlineTransducerModifiedBeamSearchDecoderAxcl
    : public OnlineTransducerDecoderAxcl {
 public:
  explicit OnlineTransducerModifiedBeamSearchDecoderAxcl(
      OnlineZipformerTransducerModelAxcl *model, int32_t max_active_paths,
      int32_t unk_id = 2, float blank_penalty = 0.0)
      : model_(model),
        max_active_paths_(max_active_paths),
        unk_id_(unk_id),
        blank_penalty_(blank_penalty) {}

  OnlineTransducerDecoderResultAxcl GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResultAxcl *r) const override;

  void Decode(std::vector<float> encoder_out,
              OnlineTransducerDecoderResultAxcl *result) const override;

 private:
  OnlineZipformerTransducerModelAxcl *model_;  // Not owned
  int32_t max_active_paths_;
  int32_t unk_id_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXCL_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_AXCL_H_
