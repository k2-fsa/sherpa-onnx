// sherpa-onnx/csrc/offline-canary-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_CANARY_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CANARY_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineCanaryModelConfig {
  std::string encoder;
  std::string decoder;

  // en, de, es, fr, or leave it empty to use en
  std::string src_lang;

  // en, de, es, fr, or leave it empty to use en
  std::string tgt_lang;

  // true to enable punctuations and casing
  // false to disable punctuations and casing
  bool use_pnc = true;

  OfflineCanaryModelConfig() = default;
  OfflineCanaryModelConfig(const std::string &encoder,
                           const std::string &decoder,
                           const std::string &src_lang,
                           const std::string &tgt_lang, bool use_pnc)
      : encoder(encoder),
        decoder(decoder),
        src_lang(src_lang),
        tgt_lang(tgt_lang),
        use_pnc(use_pnc) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CANARY_MODEL_CONFIG_H_
