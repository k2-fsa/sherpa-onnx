// sherpa-onnx/csrc/online-transducer-modified-beam-search-nemo-decoder.h
//
// Copyright (c)  2026  Joseph Mills (github.com/josephomills)

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_NEMO_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_NEMO_DECODER_H_

#include <variant>

#include "sherpa-onnx/csrc/online-transducer-nemo-decoder.h"

namespace sherpa_onnx {

class OnlineTransducerNeMoModel;
class OnlineTransducerNeMoParakeetUnifiedModel;

// Streaming port of OfflineTransducerModifiedBeamSearchNeMoDecoder.
//
// Between chunks, the active hypotheses (including a copy of the stateful
// prediction-network states per hypothesis) are kept in the `hyps` field of
// the stream's OnlineTransducerDecoderResult, so decoding resumes exactly
// where the previous chunk stopped.
class OnlineTransducerModifiedBeamSearchNeMoDecoder
    : public OnlineTransducerNeMoDecoder {
 public:
  OnlineTransducerModifiedBeamSearchNeMoDecoder(
      OnlineTransducerNeMoModel *model, int32_t max_active_paths,
      float blank_penalty)
      : model_(model),
        max_active_paths_(max_active_paths),
        blank_penalty_(blank_penalty) {}

  OnlineTransducerModifiedBeamSearchNeMoDecoder(
      OnlineTransducerNeMoParakeetUnifiedModel *model, int32_t max_active_paths,
      float blank_penalty)
      : model_(model),
        max_active_paths_(max_active_paths),
        blank_penalty_(blank_penalty) {}

  void Decode(Ort::Value encoder_out, OnlineStream **ss,
              int32_t n) const override;

 private:
  // The recognizer owns the model.
  std::variant<OnlineTransducerNeMoModel *,
               OnlineTransducerNeMoParakeetUnifiedModel *>
      model_;

  int32_t max_active_paths_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_NEMO_DECODER_H_
