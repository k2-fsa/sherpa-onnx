// sherpa-onnx/csrc/offline-transducer-modified-beam-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-model.h"

namespace sherpa_onnx {

class OfflineTransducerModifiedBeamSearchDecoder
    : public OfflineTransducerDecoder {
 public:
  OfflineTransducerModifiedBeamSearchDecoder(OfflineTransducerModel *model,
                                             int32_t max_active_paths)
      : model_(model), max_active_paths_(max_active_paths) {}

  std::vector<OfflineTransducerDecoderResult> Decode(
      Ort::Value encoder_out, Ort::Value encoder_out_length) override;

 private:
  OfflineTransducerModel *model_;  // Not owned
  int32_t max_active_paths_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_DECODER_H_
