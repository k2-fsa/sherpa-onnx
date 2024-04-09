// sherpa-onnx/csrc/audio-tagging-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/offline-zipformer-audio-tagging-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct AudioTaggingModelConfig {
  struct OfflineZipformerAudioTaggingModelConfig zipformer;

  AudioTaggingModelConfig() = default;

  explicit AudioTaggingModelConfig(
      const OfflineZipformerAudioTaggingModelConfig &zipformer)
      : zipformer(zipformer) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_MODEL_CONFIG_H_
