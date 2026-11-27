// sherpa-onnx/csrc/offline-gigaam-model-config.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/offline-gigaam-model-config.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

int32_t GigaAMLanguageSlot(const std::string &language) {
  constexpr std::array<std::string_view, 22> kLanguages = {
      "auto", "ru", "en", "kk", "ky", "uz", "be", "fr", "es", "ba", "ka",
      "hy",   "ro", "tk", "az", "tg", "de", "it", "nl", "pl", "pt", "tt"};
  std::string_view code = language == "kz" ? std::string_view("kk") : language;
  auto it = std::find(kLanguages.begin(), kLanguages.end(), code);
  return it == kLanguages.end() ? -1 : std::distance(kLanguages.begin(), it);
}

void OfflineGigaAMModelConfig::Register(ParseOptions *po) {
  po->Register("gigaam-encoder", &encoder, "Path to the GigaAM RNN-T encoder");
  po->Register("gigaam-advance", &advance, "Path to the GigaAM RNN-T advance");
  po->Register("gigaam-tokenizer", &tokenizer,
               "Path to the SentencePiece model for the selected GigaAM head");
  po->Register("gigaam-language", &language,
               "auto or language code for GigaAM Multilingual-v2");
  po->Register("gigaam-head", &head, "GigaAM RNN-T output head: e2e or raw");
}

bool OfflineGigaAMModelConfig::Validate() const {
  for (const auto *path : {&encoder, &advance, &tokenizer}) {
    if (!FileExists(*path)) {
      SHERPA_ONNX_LOGE("GigaAM file does not exist: '%s'", path->c_str());
      return false;
    }
  }

  if (head != "e2e" && head != "raw") {
    SHERPA_ONNX_LOGE("Invalid GigaAM head: '%s'", head.c_str());
    return false;
  }
  if ((head == "e2e" && EndsWith(tokenizer, "_raw_tokenizer.model")) ||
      (head == "raw" && EndsWith(tokenizer, "_e2e_tokenizer.model"))) {
    SHERPA_ONNX_LOGE("GigaAM tokenizer '%s' does not match head '%s'",
                     tokenizer.c_str(), head.c_str());
    return false;
  }
  if (GigaAMLanguageSlot(language) < 0) {
    SHERPA_ONNX_LOGE("Unknown GigaAM language: '%s'", language.c_str());
    return false;
  }
  return true;
}

std::string OfflineGigaAMModelConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineGigaAMModelConfig(encoder=\"" << encoder << "\", advance=\""
     << advance << "\", tokenizer=\"" << tokenizer << "\", language=\""
     << language << "\", head=\"" << head << "\")";
  return os.str();
}

}  // namespace sherpa_onnx
