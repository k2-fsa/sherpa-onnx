// sherpa-onnx/csrc/speaker-embedding-extractor-wespeaker-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_WESPEAKER_MODEL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_WESPEAKER_MODEL_H_

#include <memory>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/speaker-embedding-extractor-wespeaker-model-metadata.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorWeSpeakerModel {
 public:
  explicit SpeakerEmbeddingExtractorWeSpeakerModel(
      const SpeakerEmbeddingExtractorConfig &config);

  ~SpeakerEmbeddingExtractorWeSpeakerModel();

  const SpeakerEmbeddingExtractorWeSpeakerModelMetaData &GetMetaData() const;

  /**
   * @param x A float32 tensor of shape (N, T, C)
   * @return A float32 tensor of shape (N, C)
   */
  Ort::Value Compute(Ort::Value x) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_WESPEAKER_MODEL_H_
