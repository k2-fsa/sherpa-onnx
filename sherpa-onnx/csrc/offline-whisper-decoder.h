// sherpa-onnx/csrc/offline-whisper-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_DECODER_H_

#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-whisper-model-config.h"

namespace sherpa_onnx {

struct OfflineWhisperDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;
  std::string lang;
};

class OfflineWhisperDecoder {
 public:
  virtual ~OfflineWhisperDecoder() = default;

  /** Run beam search given the output from the whisper encoder model.
   *
   * @param n_layer_cross_k       A 4-D tensor of shape
   *                              (n_text_layer, N, n_audio_ctx, n_text_state).
   * @param n_layer_cross_v       A 4-D tensor of shape
   *                              (n_text_layer, N, n_audio_ctx, n_text_state).
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual std::vector<OfflineWhisperDecoderResult> Decode(
      Ort::Value n_layer_cross_k, Ort::Value n_layer_cross_v) = 0;

  virtual void SetConfig(const OfflineWhisperModelConfig &config) = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_DECODER_H_
