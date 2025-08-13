// sherpa-onnx/csrc/offline-tts-zipvoice-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsZipvoiceModelConfig {
  std::string tokens;
  std::string text_model;
  std::string flow_matching_model;
  std::string vocoder;

  // If data_dir is given, lexicon is ignored
  // data_dir is for piper-phonemize, which uses espeak-ng
  std::string data_dir;

  // Used for converting Chinese characters to pinyin
  std::string pinyin_dict;

  int num_step = 16;
  float feat_scale = 0.1;
  float speed = 1.0;
  float t_shift = 0.5;
  float target_rms = 0.1;
  float guidance_scale = 1.0;

  OfflineTtsZipvoiceModelConfig() = default;

  OfflineTtsZipvoiceModelConfig(
      const std::string &tokens, const std::string &text_model,
      const std::string &flow_matching_model, const std::string &vocoder,
      const std::string &data_dir, const std::string &pinyin_dict,
      int num_step = 16, float feat_scale = 0.1, float speed = 1.0,
      float t_shift = 0.5, float target_rms = 0.1, float guidance_scale = 1.0)
      : tokens(tokens),
        text_model(text_model),
        flow_matching_model(flow_matching_model),
        vocoder(vocoder),
        data_dir(data_dir),
        pinyin_dict(pinyin_dict),
        num_step(num_step),
        feat_scale(feat_scale),
        speed(speed),
        t_shift(t_shift),
        target_rms(target_rms),
        guidance_scale(guidance_scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_CONFIG_H_
