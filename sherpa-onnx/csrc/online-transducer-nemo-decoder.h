// sherpa-onnx/csrc/online-transducer-nemo-decoder.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_DECODER_H_

#include <cstdint>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

class OnlineStream;

// Common interface of decoders for streaming NeMo transducer models,
// whose prediction network is stateful. Decoding state is kept on the
// stream, so implementations receive the streams themselves.
class OnlineTransducerNeMoDecoder {
 public:
  virtual ~OnlineTransducerNeMoDecoder() = default;

  /** Run decoding given the output from the encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (batch_size, T, encoder_out_dim)
   * @param ss The streams corresponding to rows of encoder_out.
   * @param n Number of elements in ss.
   */
  virtual void Decode(Ort::Value encoder_out, OnlineStream **ss,
                      int32_t n) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_DECODER_H_
