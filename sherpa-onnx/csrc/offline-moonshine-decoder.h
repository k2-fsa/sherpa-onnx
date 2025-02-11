// sherpa-onnx/csrc/offline-moonshine-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_DECODER_H_

#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

struct OfflineMoonshineDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;
};

class OfflineMoonshineDecoder {
 public:
  virtual ~OfflineMoonshineDecoder() = default;

  /** Run beam search given the output from the moonshine encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (batch_size, T, dim)
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual std::vector<OfflineMoonshineDecoderResult> Decode(
      Ort::Value encoder_out) = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_DECODER_H_
