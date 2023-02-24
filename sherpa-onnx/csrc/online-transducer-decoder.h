// sherpa-onnx/csrc/online-transducer-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_DECODER_H_

#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/hypothesis.h"

namespace sherpa_onnx {

struct OnlineTransducerDecoderResult {
  /// The decoded token IDs so far
  std::vector<int64_t> tokens;

  /// number of trailing blank frames decoded so far
  int32_t num_trailing_blanks = 0;

  // used only in modified beam_search
  Hypotheses hyps;
};

class OnlineTransducerDecoder {
 public:
  virtual ~OnlineTransducerDecoder() = default;

  /* Return an empty result.
   *
   * To simplify the decoding code, we add `context_size` blanks
   * to the beginning of the decoding result, which will be
   * stripped by calling `StripPrecedingBlanks()`.
   */
  virtual OnlineTransducerDecoderResult GetEmptyResult() const = 0;

  /** Strip blanks added by `GetEmptyResult()`.
   *
   * @param r It is changed in-place.
   */
  virtual void StripLeadingBlanks(OnlineTransducerDecoderResult * /*r*/) const {
  }

  /** Run transducer beam search given the output from the encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (N, T, joiner_dim)
   * @param result  It is modified in-place.
   *
   * @note There is no need to pass encoder_out_length here since for the
   * online decoding case, each utterance has the same number of frames
   * and there are no paddings.
   */
  virtual void Decode(Ort::Value encoder_out,
                      std::vector<OnlineTransducerDecoderResult> *result) = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_DECODER_H_
