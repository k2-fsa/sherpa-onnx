// sherpa-onnx/csrc/speaker-embedding-extractor-wespeaker-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_WESPEAKER_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_WESPEAKER_IMPL_H_
#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorWeSpeakerImpl
    : public SpeakerEmbeddingExtractorImpl {
 public:
  explicit SpeakerEmbeddingExtractorWeSpeakerImpl(
      const SpeakerEmbeddingExtractorConfig &config) {}

  int32_t Dim() const override { return 0; }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    return nullptr;
  }

  std::vector<float> Compute(OnlineStream *s) const override { return {}; }

 private:
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_WESPEAKER_IMPL_H_
