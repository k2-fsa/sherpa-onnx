// sherpa-onnx/csrc/piper-phonemize.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/piper-phonemize.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/piper-voice.h"
#include "phonemize.hpp"

// Forward declaration from sherpa-onnx
namespace sherpa_onnx {
void CallPhonemizeEspeak(const std::string &text,
                         ::piper::eSpeakPhonemeConfig &config,
                         std::vector<std::vector<::piper::Phoneme>> *phonemes);
}

namespace sherpa_onnx {
namespace piper {

// language -> phoneme -> [phoneme, ...]
std::map<std::string, PhonemeMap> DEFAULT_PHONEME_MAP = {
    {"pt-br", {{U'c', {U'k'}}}}
};

// Simple text to case conversion (basic implementation)
std::string to_lowercase_utf8(const std::string& text) {
  std::string result = text;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}

std::string to_casefold_utf8(const std::string& text) {
  return to_lowercase_utf8(text); // Simplified version
}

// Simple UTF-8 to UTF-32 conversion for basic cases
std::vector<char32_t> utf8_to_utf32(const std::string& text) {
  std::vector<char32_t> result;
  
  for (size_t i = 0; i < text.length(); ) {
    unsigned char c = text[i];
    char32_t codepoint = 0;
    
    if (c < 0x80) {
      // ASCII
      codepoint = c;
      i++;
    } else if ((c & 0xE0) == 0xC0) {
      // 2-byte sequence
      if (i + 1 < text.length()) {
        codepoint = ((c & 0x1F) << 6) | (text[i + 1] & 0x3F);
        i += 2;
      } else {
        i++;
      }
    } else if ((c & 0xF0) == 0xE0) {
      // 3-byte sequence
      if (i + 2 < text.length()) {
        codepoint = ((c & 0x0F) << 12) | 
                   ((text[i + 1] & 0x3F) << 6) | 
                   (text[i + 2] & 0x3F);
        i += 3;
      } else {
        i++;
      }
    } else if ((c & 0xF8) == 0xF0) {
      // 4-byte sequence
      if (i + 3 < text.length()) {
        codepoint = ((c & 0x07) << 18) | 
                   ((text[i + 1] & 0x3F) << 12) |
                   ((text[i + 2] & 0x3F) << 6) | 
                   (text[i + 3] & 0x3F);
        i += 4;
      } else {
        i++;
      }
    } else {
      // Invalid UTF-8
      i++;
    }
    
    if (codepoint > 0) {
      result.push_back(codepoint);
    }
  }
  
  return result;
}

// Text to phonemes conversion using espeak-ng
bool phonemize(const std::string& text, const PhonemizeConfig& config,
               std::vector<std::vector<Phoneme>>& phonemes) {
  
  // Use sherpa-onnx's espeak-ng integration for phonemization
  
  // Create espeak configuration from voice config
  ::piper::eSpeakPhonemeConfig espeak_config;
  espeak_config.voice = config.voice; // Voice from loaded JSON config
  
  // Apply phoneme mapping if available
  if (config.phonemeMap) {
    espeak_config.phonemeMap = config.phonemeMap;
  }
  
  SHERPA_ONNX_LOGE("Using espeak-ng voice from loaded config: %s", espeak_config.voice.c_str());
  
  try {
    // Call sherpa-onnx's espeak phonemize function
    sherpa_onnx::CallPhonemizeEspeak(text, espeak_config, &phonemes);
    
    SHERPA_ONNX_LOGE("Phonemized text into %zu sentence(s)", phonemes.size());
    
    // Apply post-processing based on configuration
    for (auto& sentence : phonemes) {
      // Apply punctuation markers based on config
      for (auto& phoneme : sentence) {
        // Apply phoneme substitutions based on config
        if (phoneme == U'.') phoneme = config.period;
        else if (phoneme == U',') phoneme = config.comma;
        else if (phoneme == U'?') phoneme = config.question;
        else if (phoneme == U'!') phoneme = config.exclamation;
        else if (phoneme == U':') phoneme = config.colon;
        else if (phoneme == U';') phoneme = config.semicolon;
        else if (phoneme == U' ') phoneme = config.space;
      }
      
      // Remove language flags if requested
      if (!config.keepLanguageFlags) {
        // Remove patterns like "(en)" from phoneme sequence
        auto it = sentence.begin();
        while (it != sentence.end()) {
          if (*it == U'(' && (it + 3) < sentence.end() && *(it + 3) == U')') {
            // Check if it's a language flag pattern
            char32_t lang1 = *(it + 1);
            char32_t lang2 = *(it + 2);
            if ((lang1 >= U'a' && lang1 <= U'z') && (lang2 >= U'a' && lang2 <= U'z')) {
              // Remove the 4-character language flag
              it = sentence.erase(it, it + 4);
              continue;
            }
          }
          ++it;
        }
      }
    }
    
    return true;
    
  } catch (const std::exception& e) {
    SHERPA_ONNX_LOGE("Failed to phonemize text with espeak-ng: %s", e.what());
    return false;
  } catch (...) {
    SHERPA_ONNX_LOGE("Unknown error in espeak-ng phonemization");
    return false;
  }
}

void phonemize_codepoints(const std::string& text, CodepointsPhonemeConfig& config,
                         std::vector<std::vector<Phoneme>>& phonemes) {
  
  std::string processedText = text;
  
  if (config.casing == CASING_LOWER) {
    processedText = to_lowercase_utf8(processedText);
  } else if (config.casing == CASING_FOLD) {
    processedText = to_casefold_utf8(processedText);
  }

  // Convert to UTF-32 codepoints
  auto codepoints = utf8_to_utf32(processedText);

  // No sentence boundary detection
  phonemes.emplace_back();
  auto sentPhonemes = &phonemes[phonemes.size() - 1];

  if (config.phonemeMap) {
    for (auto phoneme : codepoints) {
      if (config.phonemeMap->count(phoneme) < 1) {
        // No mapping for phoneme
        sentPhonemes->push_back(phoneme);
      } else {
        // Mapping for phoneme
        auto mappedPhonemes = &(config.phonemeMap->at(phoneme));
        sentPhonemes->insert(sentPhonemes->end(), 
                            mappedPhonemes->begin(),
                            mappedPhonemes->end());
      }
    }
  } else {
    // No phoneme map
    sentPhonemes->insert(sentPhonemes->end(), 
                        codepoints.begin(), 
                        codepoints.end());
  }
}

}  // namespace piper
}  // namespace sherpa_onnx