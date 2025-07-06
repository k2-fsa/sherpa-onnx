// sherpa-onnx/csrc/offline-canary-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-canary-model-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineCanaryModelConfig::Register(ParseOptions *po) {
  po->Register("canary-encoder", &encoder,
               "Path to onnx encoder of Canary, e.g., encoder.int8.onnx");

  po->Register("canary-decoder", &decoder,
               "Path to onnx decoder of Canary, e.g., decoder.int8.onnx");

  po->Register("canary-src-lang", &src_lang,
               "Valid values: en, de, es, fr. If empty, default to use en");

  po->Register("canary-tgt-lang", &tgt_lang,
               "Valid values: en, de, es, fr. If empty, default to use en");

  po->Register("canary-use-pnc", &use_pnc,
               "true to enable punctuations and casing. false to disable them");
}

bool OfflineCanaryModelConfig::Validate() const {
  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --canary-encoder");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("Canary encoder file '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --canary-decoder");
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("Canary decoder file '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  if (!src_lang.empty()) {
    if (src_lang != "en" && src_lang != "de" && src_lang != "es" &&
        src_lang != "fr") {
      SHERPA_ONNX_LOGE("Please use en, de, es, or fr for --canary-src-lang");
      return false;
    }
  }

  if (!tgt_lang.empty()) {
    if (tgt_lang != "en" && tgt_lang != "de" && tgt_lang != "es" &&
        tgt_lang != "fr") {
      SHERPA_ONNX_LOGE("Please use en, de, es, or fr for --canary-tgt-lang");
      return false;
    }
  }

  return true;
}

std::string OfflineCanaryModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineCanaryModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "src_lang=\"" << src_lang << "\", ";
  os << "tgt_lang=\"" << tgt_lang << "\", ";
  os << "use_pnc=" << (use_pnc ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_onnx
