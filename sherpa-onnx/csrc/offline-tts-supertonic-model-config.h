// sherpa-onnx/csrc/offline-tts-supertonic-model-config.h
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsSupertonicModelConfig {
  // Individual model file paths
  std::string duration_predictor;
  std::string text_encoder;
  std::string vector_estimator;
  std::string vocoder;

  // Path to TTS config file (binary; sample_rate, base_chunk_size, etc.)
  std::string tts_config;

  // Path to unicode_indexer.bin (raw int32 array)
  std::string unicode_indexer;

  // Path to voice.bin
  std::string voice_style;

  OfflineTtsSupertonicModelConfig() = default;

  OfflineTtsSupertonicModelConfig(const std::string &duration_predictor,
                                  const std::string &text_encoder,
                                  const std::string &vector_estimator,
                                  const std::string &vocoder,
                                  const std::string &tts_config,
                                  const std::string &unicode_indexer,
                                  const std::string &voice_style)
      : duration_predictor(duration_predictor),
        text_encoder(text_encoder),
        vector_estimator(vector_estimator),
        vocoder(vocoder),
        tts_config(tts_config),
        unicode_indexer(unicode_indexer),
        voice_style(voice_style) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_SUPERTONIC_MODEL_CONFIG_H_
