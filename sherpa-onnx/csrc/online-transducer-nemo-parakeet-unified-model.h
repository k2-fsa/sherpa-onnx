// sherpa-onnx/csrc/online-transducer-nemo-parakeet-unified-model.h
//
// Copyright (c)  2026  Milan Leonard

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_PARAKEET_UNIFIED_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_PARAKEET_UNIFIED_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"

namespace sherpa_onnx {

class OnlineTransducerNeMoParakeetUnifiedModel {
 public:
  explicit OnlineTransducerNeMoParakeetUnifiedModel(
      const OnlineModelConfig &config);

  template <typename Manager>
  OnlineTransducerNeMoParakeetUnifiedModel(Manager *mgr,
                                           const OnlineModelConfig &config);

  ~OnlineTransducerNeMoParakeetUnifiedModel();

  /** Run the encoder.
   *
   * @param features A tensor of shape (N, T, C). It is changed in-place.
   * @param features_length A tensor of shape (N,) with dtype int64_t.
   *
   * @return Return encoder outputs from the ONNX encoder. ans[0] has shape
   *         (N, encoder_dim, T').
   */
  std::vector<Ort::Value> RunEncoder(Ort::Value features,
                                     Ort::Value features_length) const;

  std::pair<Ort::Value, std::vector<Ort::Value>> RunDecoder(
      Ort::Value targets, std::vector<Ort::Value> states) const;

  std::vector<Ort::Value> GetDecoderInitStates() const;

  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out) const;

  int32_t LeftFeatureFrames() const;
  int32_t ChunkFeatureFrames() const;
  int32_t RightFeatureFrames() const;
  int32_t TotalFeatureFrames() const;

  int32_t LeftEncoderFrames() const;
  int32_t ChunkEncoderFrames() const;
  int32_t RightEncoderFrames() const;

  int32_t SubsamplingFactor() const;
  int32_t VocabSize() const;
  int32_t FeatureDim() const;

  OrtAllocator *Allocator() const;

  std::string FeatureNormalizationMethod() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_NEMO_PARAKEET_UNIFIED_MODEL_H_
