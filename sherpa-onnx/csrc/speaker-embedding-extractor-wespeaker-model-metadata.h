// sherpa-onnx/csrc/speaker-embedding-extractor-wespeaker-model-metadata.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_WESPEAKER_MODEL_METADATA_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_WESPEAKER_MODEL_METADATA_H_

#include <cstdint>
#include <string>

namespace sherpa_onnx {

struct SpeakerEmbeddingExtractorWeSpeakerModelMetaData {
  int32_t output_dim = 0;
  int32_t sample_rate = 0;
  int32_t normalize_samples = 0;
  std::string language;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_WESPEAKER_MODEL_METADATA_H_
