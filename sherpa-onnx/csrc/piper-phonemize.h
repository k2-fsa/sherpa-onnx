// sherpa-onnx/csrc/piper-phonemize.h
//
// Copyright (c)  2025  Xiaomi Corporation
// Adapted from Piper TTS phonemize implementation

#ifndef SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_H_
#define SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {
namespace piper {

typedef char32_t Phoneme;
typedef std::map<Phoneme, std::vector<Phoneme>> PhonemeMap;

// Forward declaration
struct PhonemizeConfig;

struct PiperPhonemeConfig {
  std::string voice = "en-us";

  Phoneme period = U'.';      // CLAUSE_PERIOD
  Phoneme comma = U',';       // CLAUSE_COMMA
  Phoneme question = U'?';    // CLAUSE_QUESTION
  Phoneme exclamation = U'!'; // CLAUSE_EXCLAMATION
  Phoneme colon = U':';       // CLAUSE_COLON
  Phoneme semicolon = U';';   // CLAUSE_SEMICOLON
  Phoneme space = U' ';

  // Remove language switch flags like "(en)"
  bool keepLanguageFlags = false;

  std::shared_ptr<PhonemeMap> phonemeMap;
};

enum TextCasing {
  CASING_IGNORE = 0,
  CASING_LOWER = 1,
  CASING_UPPER = 2,
  CASING_FOLD = 3
};

// Configuration for phonemize_codepoints
struct CodepointsPhonemeConfig {
  TextCasing casing = CASING_FOLD;
  std::shared_ptr<PhonemeMap> phonemeMap;
};

// Real phonemize implementation using espeak-ng (equivalent to UE plugin)
bool phonemize(const std::string& text, const PhonemizeConfig& config,
               std::vector<std::vector<Phoneme>>& phonemes);

void phonemize_codepoints(const std::string& text, CodepointsPhonemeConfig& config,
                         std::vector<std::vector<Phoneme>>& phonemes);

}  // namespace piper
}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_H_