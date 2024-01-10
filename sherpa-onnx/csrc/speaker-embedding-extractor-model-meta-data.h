// sherpa-onnx/csrc/speaker-embedding-extractor-model-metadata.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_META_DATA_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_META_DATA_H_

#include <cstdint>
#include <string>

namespace sherpa_onnx {

struct SpeakerEmbeddingExtractorModelMetaData {
  int32_t output_dim = 0;
  int32_t sample_rate = 0;

  // for wespeaker models, it is 0;
  // for 3d-speaker models, it is 1
  int32_t normalize_samples = 0;

  // Chinese, English, etc.
  std::string language;

  // for 3d-speaker, it is global-mean
  std::string feature_normalize_type;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_MODEL_META_DATA_H_
