// sherpa-onnx/csrc/rknn/online-transducer-decoder-rknn.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_ONLINE_TRANSDUCER_DECODER_RKNN_H_
#define SHERPA_ONNX_CSRC_RKNN_ONLINE_TRANSDUCER_DECODER_RKNN_H_

#include <vector>

#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

struct OnlineTransducerDecoderResultRknn {
  /// Number of frames after subsampling we have decoded so far
  int32_t frame_offset = 0;

  /// The decoded token IDs so far
  std::vector<int64_t> tokens;

  /// number of trailing blank frames decoded so far
  int32_t num_trailing_blanks = 0;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  std::vector<int32_t> timestamps;

  // used only by greedy_search
  std::vector<float> previous_decoder_out;

  // used only in modified beam_search
  Hypotheses hyps;

  // used only by modified_beam_search
  std::vector<std::vector<float>> previous_decoder_out2;
};

class OnlineTransducerDecoderRknn {
 public:
  virtual ~OnlineTransducerDecoderRknn() = default;

  /* Return an empty result.
   *
   * To simplify the decoding code, we add `context_size` blanks
   * to the beginning of the decoding result, which will be
   * stripped by calling `StripPrecedingBlanks()`.
   */
  virtual OnlineTransducerDecoderResultRknn GetEmptyResult() const = 0;

  /** Strip blanks added by `GetEmptyResult()`.
   *
   * @param r It is changed in-place.
   */
  virtual void StripLeadingBlanks(
      OnlineTransducerDecoderResultRknn * /*r*/) const {}

  virtual void Decode(std::vector<float> encoder_out,
                      OnlineTransducerDecoderResultRknn *result) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_ONLINE_TRANSDUCER_DECODER_RKNN_H_
