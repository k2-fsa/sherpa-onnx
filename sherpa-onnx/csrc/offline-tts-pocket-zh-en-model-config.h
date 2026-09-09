// sherpa-onnx/csrc/offline-tts-pocket-zh-en-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Please refer to
// https://modelscope.cn/models/dengcunqin/pocket-tts-zh-en
// for the model files.

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsPocketZhEnModelConfig {
  std::string step_model;    // Path to step_model.onnx
  std::string step_encoder;  // Path to step_encoder.onnx
  std::string lexicon;       // Path to lexicon.txt

  int32_t voice_embedding_cache_capacity = 50;

  OfflineTtsPocketZhEnModelConfig() = default;

  OfflineTtsPocketZhEnModelConfig(const std::string &step_model,
                                  const std::string &step_encoder,
                                  const std::string &lexicon,
                                  int32_t voice_embedding_cache_capacity)
      : step_model(step_model),
        step_encoder(step_encoder),
        lexicon(lexicon),
        voice_embedding_cache_capacity(voice_embedding_cache_capacity) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_MODEL_CONFIG_H_
