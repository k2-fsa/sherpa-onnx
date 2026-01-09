// sherpa-onnx/csrc/offline-whisper-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-model-config.h"

#include <string>
#include <unordered_map>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineWhisperModelConfig::Register(ParseOptions *po) {
  po->Register("whisper-encoder", &encoder,
               "Path to onnx encoder of whisper, e.g., tiny-encoder.onnx, "
               "medium.en-encoder.onnx.");

  po->Register("whisper-decoder", &decoder,
               "Path to onnx decoder of whisper, e.g., tiny-decoder.onnx, "
               "medium.en-decoder.onnx.");

  po->Register(
      "whisper-language", &language,
      "The spoken language in the input audio file. Example values: "
      "en, de, fr, zh, jp. If it is not given for a multilingual model, we will"
      " infer the language from the input audio file. "
      "Please refer to "
      "https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10"
      " for valid values. Note that for non-multilingual models, it supports "
      "only 'en'");

  po->Register("whisper-task", &task,
               "Valid values: transcribe, translate. "
               "Note that for non-multilingual models, it supports "
               "only 'transcribe'");

  po->Register(
      "whisper-tail-paddings", &tail_paddings,
      "Suggested value: 50 for English models. 300 for multilingual models. "
      "Since we have removed the 30-second constraint, we need to add some "
      "tail padding frames "
      "so that whisper can detect the eot token. Leave it to -1 to use 1000.");
}

bool OfflineWhisperModelConfig::Validate() const {
  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --whisper-encoder");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("whisper encoder file '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --whisper-decoder");
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("whisper decoder file '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  if (task != "translate" && task != "transcribe") {
    SHERPA_ONNX_LOGE(
        "--whisper-task supports only translate and transcribe. Given: %s",
        task.c_str());

    return false;
  }

  return true;
}

std::string OfflineWhisperModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineWhisperModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "language=\"" << language << "\", ";
  os << "task=\"" << task << "\", ";
  os << "tail_paddings=" << tail_paddings << ")";

  return os.str();
}

bool IsMultilingual(WhisperModelType model_type) {
  switch (model_type) {
    case WhisperModelType::TinyEn:
    case WhisperModelType::BaseEn:
    case WhisperModelType::SmallEn:
    case WhisperModelType::MediumEn:
      return false;  // English-only models

    case WhisperModelType::Tiny:
    case WhisperModelType::Base:
    case WhisperModelType::Small:
    case WhisperModelType::Medium:
    case WhisperModelType::Large:
      return true;  // Multilingual models
  }

  SHERPA_ONNX_LOGE("Unsupported model: %s", ToString(model_type).c_str());
  SHERPA_ONNX_EXIT(-1);
  // Safety fallback (should never be hit)
  return false;
}

std::string ToString(WhisperModelType model) {
  switch (model) {
    case WhisperModelType::Tiny:
      return "tiny";
    case WhisperModelType::TinyEn:
      return "tiny.en";
    case WhisperModelType::Base:
      return "base";
    case WhisperModelType::BaseEn:
      return "base.en";
    case WhisperModelType::Small:
      return "small";
    case WhisperModelType::SmallEn:
      return "small.en";
    case WhisperModelType::Medium:
      return "medium";
    case WhisperModelType::MediumEn:
      return "medium.en";
    case WhisperModelType::Large:
      return "large";
  }
  return "unknown";
}

WhisperModelType ParseWhisperModelType(const std::string &name) {
  if (name == "tiny") return WhisperModelType::Tiny;
  if (name == "tiny.en") return WhisperModelType::TinyEn;
  if (name == "base") return WhisperModelType::Base;
  if (name == "base.en") return WhisperModelType::BaseEn;
  if (name == "small") return WhisperModelType::Small;
  if (name == "small.en") return WhisperModelType::SmallEn;
  if (name == "medium") return WhisperModelType::Medium;
  if (name == "medium.en") return WhisperModelType::MediumEn;
  if (name == "large") return WhisperModelType::Large;

  SHERPA_ONNX_LOGE("Unknown Whisper model: '%s'", name.c_str());
  SHERPA_ONNX_EXIT(-1);

  // Unreachable code
  return WhisperModelType::Tiny;
}

int32_t GetWhisperLanguageTokenId(const std::string &lang) {
  static const std::unordered_map<std::string, int32_t> kLangToToken = {
      {"hi", 50276},  {"cy", 50297}, {"oc", 50328}, {"so", 50326},
      {"fr", 50265},  {"az", 50304}, {"eu", 50310}, {"ba", 50355},
      {"no", 50288},  {"as", 50350}, {"nl", 50271}, {"bn", 50302},
      {"es", 50262},  {"ml", 50296}, {"km", 50323}, {"mk", 50308},
      {"sq", 50317},  {"mt", 50343}, {"et", 50307}, {"ms", 50282},
      {"tr", 50268},  {"bg", 50292}, {"ps", 50340}, {"br", 50309},
      {"ht", 50339},  {"tt", 50351}, {"tk", 50341}, {"la", 50294},
      {"de", 50261},  {"ur", 50290}, {"ro", 50284}, {"fa", 50300},
      {"uk", 50280},  {"mg", 50349}, {"lo", 50336}, {"sr", 50303},
      {"yo", 50325},  {"id", 50275}, {"da", 50285}, {"pt", 50267},
      {"nn", 50342},  {"sn", 50324}, {"sa", 50344}, {"sd", 50332},
      {"gl", 50319},  {"ja", 50266}, {"pl", 50269}, {"ru", 50263},
      {"ko", 50264},  {"ne", 50313}, {"kn", 50306}, {"zh", 50260},
      {"be", 50330},  {"ca", 50270}, {"el", 50281}, {"it", 50274},
      {"hu", 50286},  {"lt", 50293}, {"ta", 50287}, {"is", 50311},
      {"jw", 50356},  {"fi", 50277}, {"bo", 50347}, {"sv", 50273},
      {"mi", 50295},  {"hr", 50291}, {"bs", 50315}, {"yi", 50335},
      {"sk", 50298},  {"lv", 50301}, {"af", 50327}, {"vi", 50278},
      {"ha", 50354},  {"mn", 50314}, {"cs", 50283}, {"sl", 50305},
      {"pa", 50321},  {"su", 50357}, {"ka", 50329}, {"ln", 50353},
      {"lb", 50345},  {"sw", 50318}, {"en", 50259}, {"tl", 50348},
      {"hy", 50312},  {"te", 50299}, {"he", 50279}, {"my", 50346},
      {"haw", 50352}, {"fo", 50338}, {"kk", 50316}, {"si", 50322},
      {"tg", 50331},  {"th", 50289}, {"ar", 50272}, {"am", 50334},
      {"mr", 50320},  {"uz", 50337}, {"gu", 50333}};

  auto it = kLangToToken.find(lang);

  return (it != kLangToToken.end()) ? it->second : -1;
}

}  // namespace sherpa_onnx
