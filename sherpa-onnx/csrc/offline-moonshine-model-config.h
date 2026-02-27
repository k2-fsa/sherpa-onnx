// sherpa-onnx/csrc/offline-moonshine-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineMoonshineModelConfig {
  // For moonshine v1, it has 4 models:
  // preprocessor, encoder, uncached_decoder, cached_decoder
  //
  // For moonshine v2, it has 2 models:
  // encoder, merged_decoder
  //
  // You can choose either v1 by providing 4 models or
  // select v2 by providing 2 models, but not both

  std::string preprocessor;
  std::string encoder;
  std::string uncached_decoder;
  std::string cached_decoder;

  std::string merged_decoder;

  OfflineMoonshineModelConfig() = default;
  OfflineMoonshineModelConfig(const std::string &preprocessor,
                              const std::string &encoder,
                              const std::string &uncached_decoder,
                              const std::string &cached_decoder,
                              const std::string &merged_decoder)
      : preprocessor(preprocessor),
        encoder(encoder),
        uncached_decoder(uncached_decoder),
        cached_decoder(cached_decoder),
        merged_decoder(merged_decoder) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MOONSHINE_MODEL_CONFIG_H_
