// sherpa-onnx/csrc/offline-tts-cache-mechanism-config.h
//
// Copyright (c)  2025  @mah92 From Iranian people to the community with love

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CACHE_MECHANISM_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CACHE_MECHANISM_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsCacheMechanismConfig {

  std::string cache_dir;

  int32_t cache_size;

  OfflineTtsCacheMechanismConfig() = default;

  OfflineTtsCacheMechanismConfig(const std::string &cache_dir,
                                  int32_t cache_size)
      : cache_dir(cache_dir),
        cache_size(cache_size) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CACHE_MECHANISM_CONFIG_H_
