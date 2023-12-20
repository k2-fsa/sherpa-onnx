// sherpa-onnx/csrc/online-ctc-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_DECODER_H_

#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

struct OnlineCtcDecoderResult {
  /// Number of frames after subsampling we have decoded so far
  int32_t frame_offset = 0;

  /// The decoded token IDs
  std::vector<int64_t> tokens;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  /// Note: The index is after subsampling
  std::vector<int32_t> timestamps;

  int32_t num_trailing_blanks = 0;
};

class OnlineCtcDecoder {
 public:
  virtual ~OnlineCtcDecoder() = default;

  /** Run streaming CTC decoding given the output from the encoder model.
   *
   * @param log_probs A 3-D tensor of shape (N, T, vocab_size) containing
   *                  lob_probs.
   *
   * @param  results Input & Output parameters..
   */
  virtual void Decode(Ort::Value log_probs,
                      std::vector<OnlineCtcDecoderResult> *results) = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_DECODER_H_
