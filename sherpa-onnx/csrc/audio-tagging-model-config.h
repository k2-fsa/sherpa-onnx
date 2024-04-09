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

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  AudioTaggingModelConfig() = default;

  AudioTaggingModelConfig(
      const OfflineZipformerAudioTaggingModelConfig &zipformer,
      int32_t num_threads, bool debug, const std::string &provider)
      : zipformer(zipformer),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_MODEL_CONFIG_H_
