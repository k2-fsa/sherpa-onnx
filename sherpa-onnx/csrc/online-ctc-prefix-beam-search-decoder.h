// sherpa-onnx/csrc/online-ctc-prefix-beam-search-decoder.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/online-ctc-decoder.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

class OnlineCtcPrefixBeamSearchDecoder : public OnlineCtcDecoder {
 public:
  OnlineCtcPrefixBeamSearchDecoder(int32_t max_active_paths, int32_t blank_id)
      : max_active_paths_(max_active_paths), blank_id_(blank_id) {}

  void Decode(const float *log_probs, int32_t batch_size, int32_t num_frames,
              int32_t vocab_size, std::vector<OnlineCtcDecoderResult> *results,
              OnlineStream **ss = nullptr, int32_t n = 0) override;

 private:
  int32_t max_active_paths_;
  int32_t blank_id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_
