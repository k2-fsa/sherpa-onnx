// sherpa-onnx/csrc/offline-canary-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-canary-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineCanaryModelConfig::Register(ParseOptions *po) {
  po->Register("canary-encoder", &encoder,
               "Path to onnx encoder of Canary, e.g., encoder.int8.onnx");

  po->Register("canary-decoder", &decoder,
               "Path to onnx decoder of Canary, e.g., decoder.int8.onnx");

  po->Register("canary-src-lang", &src_lang,
               "Any 2-letter language code carried by the model's tokens.txt ""(en, es, de, fr for canary-180m-flash; multilingual exports such ""as canary-1b-v2 carry more). Unknown codes warn and fall back to ""en. If empty, defaults to en");

  po->Register("canary-tgt-lang", &tgt_lang,
               "Any 2-letter language code carried by the model's tokens.txt ""(en, es, de, fr for canary-180m-flash; multilingual exports such ""as canary-1b-v2 carry more). Unknown codes warn and fall back to ""en. If empty, defaults to en");

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
