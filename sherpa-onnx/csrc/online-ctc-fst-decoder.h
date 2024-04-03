// sherpa-onnx/csrc/online-ctc-fst-decoder.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_FST_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_FST_DECODER_H_

#include <memory>
#include <vector>

#include "fst/fst.h"
#include "sherpa-onnx/csrc/online-ctc-decoder.h"
#include "sherpa-onnx/csrc/online-ctc-fst-decoder-config.h"

namespace sherpa_onnx {

class OnlineCtcFstDecoder : public OnlineCtcDecoder {
 public:
  OnlineCtcFstDecoder(const OnlineCtcFstDecoderConfig &config,
                      int32_t blank_id);

  void Decode(Ort::Value log_probs,
              std::vector<OnlineCtcDecoderResult> *results,
              OnlineStream **ss = nullptr, int32_t n = 0) override;

  std::unique_ptr<kaldi_decoder::FasterDecoder> CreateFasterDecoder()
      const override;

 private:
  OnlineCtcFstDecoderConfig config_;
  kaldi_decoder::FasterDecoderOptions options_;

  std::unique_ptr<fst::Fst<fst::StdArc>> fst_;
  int32_t blank_id_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_FST_DECODER_H_
