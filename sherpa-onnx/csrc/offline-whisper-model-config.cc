// sherpa-onnx/csrc/offline-whisper-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-model-config.h"

#include <string>
#include <unordered_map>
#include <vector>

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

std::string GetWhisperLanguageCode(int32_t token_id) {
  static const std::unordered_map<int32_t, std::string> kTokenToLang = {
      {50276, "hi"},  {50297, "cy"}, {50328, "oc"}, {50326, "so"},
      {50265, "fr"},  {50304, "az"}, {50310, "eu"}, {50355, "ba"},
      {50288, "no"},  {50350, "as"}, {50271, "nl"}, {50302, "bn"},
      {50262, "es"},  {50296, "ml"}, {50323, "km"}, {50308, "mk"},
      {50317, "sq"},  {50343, "mt"}, {50307, "et"}, {50282, "ms"},
      {50268, "tr"},  {50292, "bg"}, {50340, "ps"}, {50309, "br"},
      {50339, "ht"},  {50351, "tt"}, {50341, "tk"}, {50294, "la"},
      {50261, "de"},  {50290, "ur"}, {50284, "ro"}, {50300, "fa"},
      {50280, "uk"},  {50349, "mg"}, {50336, "lo"}, {50303, "sr"},
      {50325, "yo"},  {50275, "id"}, {50285, "da"}, {50267, "pt"},
      {50342, "nn"},  {50324, "sn"}, {50344, "sa"}, {50332, "sd"},
      {50319, "gl"},  {50266, "ja"}, {50269, "pl"}, {50263, "ru"},
      {50264, "ko"},  {50313, "ne"}, {50306, "kn"}, {50260, "zh"},
      {50330, "be"},  {50270, "ca"}, {50281, "el"}, {50274, "it"},
      {50286, "hu"},  {50293, "lt"}, {50287, "ta"}, {50311, "is"},
      {50356, "jw"},  {50277, "fi"}, {50347, "bo"}, {50273, "sv"},
      {50295, "mi"},  {50291, "hr"}, {50315, "bs"}, {50335, "yi"},
      {50298, "sk"},  {50301, "lv"}, {50327, "af"}, {50278, "vi"},
      {50354, "ha"},  {50314, "mn"}, {50283, "cs"}, {50305, "sl"},
      {50321, "pa"},  {50357, "su"}, {50329, "ka"}, {50353, "ln"},
      {50345, "lb"},  {50318, "sw"}, {50259, "en"}, {50348, "tl"},
      {50312, "hy"},  {50299, "te"}, {50279, "he"}, {50346, "my"},
      {50352, "haw"}, {50338, "fo"}, {50316, "kk"}, {50322, "si"},
      {50331, "tg"},  {50289, "th"}, {50272, "ar"}, {50334, "am"},
      {50320, "mr"},  {50337, "uz"}, {50333, "gu"}};

  auto it = kTokenToLang.find(token_id);
  return (it != kTokenToLang.end()) ? it->second : std::string{};
}

const std::vector<int32_t> &GetAllWhisperLanguageTokenIds() {
  static const std::vector<int32_t> kLanguageTokenIds = {
      50276, 50297, 50328, 50326, 50265, 50304, 50310, 50355, 50288, 50350,
      50271, 50302, 50262, 50296, 50323, 50308, 50317, 50343, 50307, 50282,
      50268, 50292, 50340, 50309, 50339, 50351, 50341, 50294, 50261, 50290,
      50284, 50300, 50280, 50349, 50336, 50303, 50325, 50275, 50285, 50267,
      50342, 50324, 50344, 50332, 50319, 50266, 50269, 50263, 50264, 50313,
      50306, 50260, 50330, 50270, 50281, 50274, 50286, 50293, 50287, 50311,
      50356, 50277, 50347, 50273, 50295, 50291, 50315, 50335, 50298, 50301,
      50327, 50278, 50354, 50314, 50283, 50305, 50321, 50357, 50329, 50353,
      50345, 50318, 50259, 50348, 50312, 50299, 50279, 50346, 50352, 50338,
      50316, 50322, 50331, 50289, 50272, 50334, 50320, 50337, 50333};

  return kLanguageTokenIds;
}

const std::vector<std::string> &GetAllWhisperLanguageCodes() {
  static const std::vector<std::string> kLanguageCodes = {
      "hi",  "cy", "oc", "so", "fr", "az", "eu", "ba", "no", "as", "nl",
      "bn",  "es", "ml", "km", "mk", "sq", "mt", "et", "ms", "tr", "bg",
      "ps",  "br", "ht", "tt", "tk", "la", "de", "ur", "ro", "fa", "uk",
      "mg",  "lo", "sr", "yo", "id", "da", "pt", "nn", "sn", "sa", "sd",
      "gl",  "ja", "pl", "ru", "ko", "ne", "kn", "zh", "be", "ca", "el",
      "it",  "hu", "lt", "ta", "is", "jw", "fi", "bo", "sv", "mi", "hr",
      "bs",  "yi", "sk", "lv", "af", "vi", "ha", "mn", "cs", "sl", "pa",
      "su",  "ka", "ln", "lb", "sw", "en", "tl", "hy", "te", "he", "my",
      "haw", "fo", "kk", "si", "tg", "th", "ar", "am", "mr", "uz", "gu"};

  return kLanguageCodes;
}

}  // namespace sherpa_onnx
