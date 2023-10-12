// sherpa-onnx/csrc/offline-ctc-fst-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_CTC_FST_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CTC_FST_DECODER_H_

#include <memory>
#include <vector>

#include "fst/fst.h"
#include "sherpa-onnx/csrc/offline-ctc-decoder.h"
#include "sherpa-onnx/csrc/offline-ctc-fst-decoder-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

class OfflineCtcFstDecoder : public OfflineCtcDecoder {
 public:
  explicit OfflineCtcFstDecoder(const OfflineCtcFstDecoderConfig &config);

  std::vector<OfflineCtcDecoderResult> Decode(
      Ort::Value log_probs, Ort::Value log_probs_length) override;

 private:
  OfflineCtcFstDecoderConfig config_;

  std::unique_ptr<fst::Fst<fst::StdArc>> fst_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CTC_FST_DECODER_H_
