// sherpa-onnx/csrc/offline-tts-zipvoice-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_CONFIG_H_

#include <cstdint>
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsZipvoiceModelConfig {
  std::string tokens;
  std::string encoder;
  std::string decoder;
  std::string vocoder;

  std::string data_dir;
  std::string lexicon;

  float feat_scale = 0.1;
  float t_shift = 0.5;
  float target_rms = 0.1;
  float guidance_scale = 1.0;

  OfflineTtsZipvoiceModelConfig() = default;

  OfflineTtsZipvoiceModelConfig(
      const std::string &tokens, const std::string &encoder,
      const std::string &decoder, const std::string &vocoder,
      const std::string &data_dir, const std::string &lexicon,
      float feat_scale = 0.1, float t_shift = 0.5, float target_rms = 0.1,
      float guidance_scale = 1.0)
      : tokens(tokens),
        encoder(encoder),
        decoder(decoder),
        vocoder(vocoder),
        data_dir(data_dir),
        lexicon(lexicon),
        feat_scale(feat_scale),
        t_shift(t_shift),
        target_rms(target_rms),
        guidance_scale(guidance_scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_ZIPVOICE_MODEL_CONFIG_H_
