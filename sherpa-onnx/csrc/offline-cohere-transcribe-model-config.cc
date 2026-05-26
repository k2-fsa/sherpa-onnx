// sherpa-onnx/csrc/offline-cohere-transcribe-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-cohere-transcribe-model-config.h"

#include <sstream>
#include <string>
#include <unordered_set>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

bool IsValidCohereTranscribeLanguage(const std::string &language) {
  static const std::unordered_set<std::string> kSupportedLanguages = {
      "ar", "de", "el", "en", "es", "fr", "it",
      "ja", "ko", "nl", "pl", "pt", "vi", "zh"};

  return kSupportedLanguages.count(language) != 0;
}

void OfflineCohereTranscribeModelConfig::Register(ParseOptions *po) {
  po->Register("cohere-transcribe-encoder", &encoder,
               "Path to onnx encoder of Cohere Transcribe");

  po->Register("cohere-transcribe-decoder", &decoder,
               "Path to onnx decoder of Cohere Transcribe");

  po->Register(
      "cohere-transcribe-language", &language,
      "The spoken language in the input audio file. It supports 14 languages. "
      "Example values: "
      "ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, vi, zh. "
      "You have to provide exactly one language");

  po->Register("cohere-transcribe-use-punct", &use_punct,
               "true to enable punctuations. false to disable punctuations");

  po->Register(
      "cohere-transcribe-use-itn", &use_itn,
      "true to enable inverse text normalization. false to disable inverse "
      "text normalization");
}

bool OfflineCohereTranscribeModelConfig::Validate() const {
  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --cohere-transcribe-encoder");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("cohere-transcribe encoder file '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --cohere-transcribe-decoder");
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("cohere-transcribe decoder file '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  if (!language.empty()) {
    if (!IsValidCohereTranscribeLanguage(language)) {
      SHERPA_ONNX_LOGE(
          "Invalid --cohere-transcribe-language: '%s'. "
          "Supported values: ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, "
          "vi, zh",
          language.c_str());
      return false;
    }
  }

  return true;
}

std::string OfflineCohereTranscribeModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineCohereTranscribeModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "language=\"" << language << "\", ";
  os << "use_punct=" << (use_punct ? "True" : "False") << ", ";
  os << "use_itn=" << (use_itn ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_onnx
