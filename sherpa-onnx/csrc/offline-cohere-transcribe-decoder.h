// sherpa-onnx/csrc/offline-cohere-transcribe-decoder.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_DECODER_H_

#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-cohere-transcribe-model-config.h"

namespace sherpa_onnx {

struct OfflineCohereTranscribeDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;
};

class OfflineCohereTranscribeDecoder {
 public:
  virtual ~OfflineCohereTranscribeDecoder() = default;

  /** Run beam search given the output from the Cohere Transcribe encoder model.
   *
   * @param n_layer_cross_k       A 4-D tensor of shape
   *                              (num_layers, N, T, hidden_size)
   * @param n_layer_cross_v       A 4-D tensor of shape
   *                              (num_layers, N, T, hidden_size)
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual std::vector<OfflineCohereTranscribeDecoderResult> Decode(
      Ort::Value n_layer_cross_k, Ort::Value n_layer_cross_v,
      const std::vector<int64_t> &prompt, int32_t eos,
      int32_t num_feature_frames) = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_COHERE_TRANSCRIBE_DECODER_H_
