// sherpa-onnx/csrc/offline-fire-red-asr-decoder.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_DECODER_H_

#include <cstdint>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

struct OfflineFireRedAsrDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;
};

class OfflineFireRedAsrDecoder {
 public:
  virtual ~OfflineFireRedAsrDecoder() = default;

  /** Run beam search given the output from the FireRedAsr encoder model.
   *
   * @param n_layer_cross_k       A 4-D tensor of shape
   *                              (num_decoder_layers, N, T, d_model).
   * @param n_layer_cross_v       A 4-D tensor of shape
   *                              (num_decoder_layers, N, T, d_model).
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual std::vector<OfflineFireRedAsrDecoderResult> Decode(
      Ort::Value n_layer_cross_k, Ort::Value n_layer_cross_v) = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_FIRE_RED_ASR_DECODER_H_
