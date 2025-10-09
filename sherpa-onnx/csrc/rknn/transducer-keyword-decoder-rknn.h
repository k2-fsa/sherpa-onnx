// sherpa-onnx/csrc/rknn/transducer-keywords-decoder-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_TRANSDUCER_KEYWORD_DECODER_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_TRANSDUCER_KEYWORD_DECODER_RKNN_H_

#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/rknn/online-stream-rknn.h"
#include "sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.h"
#include "sherpa-onnx/csrc/transducer-keyword-decoder.h"

namespace sherpa_onnx {

class TransducerKeywordDecoderRknn {
 public:
  TransducerKeywordDecoderRknn(OnlineZipformerTransducerModelRknn *model,
                               int32_t max_active_paths,
                               int32_t num_trailing_blanks, int32_t unk_id)
      : model_(model),
        max_active_paths_(max_active_paths),
        num_trailing_blanks_(num_trailing_blanks),
        unk_id_(unk_id) {}

  TransducerKeywordResult GetEmptyResult() const;

  void Decode(std::vector<float> encoder_out, OnlineStreamRknn *s);

 private:
  OnlineZipformerTransducerModelRknn *model_;  // Not owned

  int32_t max_active_paths_;
  int32_t num_trailing_blanks_;
  int32_t unk_id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_TRANSDUCER_KEYWORD_DECODER_RKNN_H_
