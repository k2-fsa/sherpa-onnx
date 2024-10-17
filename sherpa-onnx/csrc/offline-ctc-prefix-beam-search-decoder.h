// sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-ctc-decoder.h"

namespace sherpa_onnx {

class OfflineCtcPrefixBeamSearchDecoder : public OfflineCtcDecoder {
 public:
  OfflineCtcPrefixBeamSearchDecoder(int32_t max_active_paths, int32_t blank_id)
      : max_active_paths_(max_active_paths), blank_id_(blank_id) {}

  std::vector<OfflineCtcDecoderResult> Decode(Ort::Value log_probs,
                                              Ort::Value log_probs_length,
                                              OfflineStream **ss = nullptr,
                                              int32_t n = 0) override;

 private:
  int32_t max_active_paths_;
  int32_t blank_id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_
