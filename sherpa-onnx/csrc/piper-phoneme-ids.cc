// sherpa-onnx/csrc/piper-phoneme-ids.cc
//
// Copyright (c)  2025  Xiaomi Corporation
// Adapted from Piper TTS phoneme_ids implementation

#include "sherpa-onnx/csrc/piper-phoneme-ids.h"

#include <map>
#include <string>
#include <vector>

namespace sherpa_onnx {
namespace piper {

// Default phoneme ID map (from Piper's DEFAULT_PHONEME_ID_MAP)
static PhonemeIdMap DEFAULT_PHONEME_ID_MAP = {
    {U'_', {0}},
    {U'^', {1}},
    {U'$', {2}},
    {U' ', {3}},
    {U'!', {4}},
    {U'\'', {5}},
    {U'(', {6}},
    {U')', {7}},
    {U',', {8}},
    {U'-', {9}},
    {U'.', {10}},
    {U':', {11}},
    {U';', {12}},
    {U'?', {13}},
    {U'a', {14}},
    {U'b', {15}},
    {U'c', {16}},
    {U'd', {17}},
    {U'e', {18}},
    {U'f', {19}},
    {U'h', {20}},
    {U'i', {21}},
    {U'j', {22}},
    {U'k', {23}},
    {U'l', {24}},
    {U'm', {25}},
    {U'n', {26}},
    {U'o', {27}},
    {U'p', {28}},
    {U'q', {29}},
    {U'r', {30}},
    {U's', {31}},
    {U't', {32}},
    {U'u', {33}},
    {U'v', {34}},
    {U'w', {35}},
    {U'x', {36}},
    {U'y', {37}},
    {U'z', {38}},
    {U'æ', {39}},
    {U'ç', {40}},
    {U'ð', {41}},
    {U'ø', {42}},
    {U'ħ', {43}},
    {U'ŋ', {44}},
    {U'œ', {45}},
    {U'ǀ', {46}},
    {U'ǁ', {47}},
    {U'ǂ', {48}},
    {U'ǃ', {49}},
    {U'ɐ', {50}},
    {U'ɑ', {51}},
    {U'ɒ', {52}},
    {U'ɓ', {53}},
    {U'ɔ', {54}},
    {U'ɕ', {55}},
    {U'ɖ', {56}},
    {U'ɗ', {57}},
    {U'ɘ', {58}},
    {U'ə', {59}},
    {U'ɚ', {60}},
    {U'ɛ', {61}},
    {U'ɜ', {62}},
    {U'ɞ', {63}},
    {U'ɟ', {64}},
    {U'ɠ', {65}},
    {U'ɡ', {66}},
    {U'ɢ', {67}},
    {U'ɣ', {68}},
    {U'ɤ', {69}},
    {U'ɥ', {70}},
    {U'ɦ', {71}},
    {U'ɧ', {72}},
    {U'ɨ', {73}},
    {U'ɪ', {74}},
    {U'ɫ', {75}},
    {U'ɬ', {76}},
    {U'ɭ', {77}},
    {U'ɮ', {78}},
    {U'ɯ', {79}},
    {U'ɰ', {80}},
    {U'ɱ', {81}},
    {U'ɲ', {82}},
    {U'ɳ', {83}},
    {U'ɴ', {84}},
    {U'ɵ', {85}},
    {U'ɶ', {86}},
    {U'ɸ', {87}},
    {U'ɹ', {88}},
    {U'ɺ', {89}},
    {U'ɻ', {90}},
    {U'ɽ', {91}},
    {U'ɾ', {92}},
    {U'ʀ', {93}},
    {U'ʁ', {94}},
    {U'ʂ', {95}},
    {U'ʃ', {96}},
    {U'ʄ', {97}},
    {U'ʈ', {98}},
    {U'ʉ', {99}},
    {U'ʊ', {100}},
    {U'ʋ', {101}},
    {U'ʌ', {102}},
    {U'ʍ', {103}},
    {U'ʎ', {104}},
    {U'ʏ', {105}},
    {U'ʐ', {106}},
    {U'ʑ', {107}},
    {U'ʒ', {108}},
    {U'ʔ', {109}},
    {U'ʕ', {110}},
    {U'ʘ', {111}},
    {U'ʙ', {112}},
    {U'ʛ', {113}},
    {U'ʜ', {114}},
    {U'ʝ', {115}},
    {U'ʟ', {116}},
    {U'ʡ', {117}},
    {U'ʢ', {118}},
    {U'ʲ', {119}},
    {U'ˈ', {120}},
    {U'ˌ', {121}},
    {U'ː', {122}},
    {U'ˑ', {123}},
    {U'˞', {124}},
    {U'β', {125}},
    {U'θ', {126}},
    {U'χ', {127}},
    {U'ᵻ', {128}},
    {U'ⱱ', {129}},

    // tones
    {U'0', {130}},
    {U'1', {131}},
    {U'2', {132}},
    {U'3', {133}},
    {U'4', {134}},
    {U'5', {135}},
    {U'6', {136}},
    {U'7', {137}},
    {U'8', {138}},
    {U'9', {139}},
    {U'\u0327', {140}}, // combining cedilla
    {U'\u0303', {141}}, // combining tilde
    {U'\u032a', {142}}, // combining bridge below
    {U'\u032f', {143}}, // combining inverted breve below
    {U'\u0329', {144}}, // combining vertical line below
    {U'ʰ', {145}},
    {U'ˤ', {146}},
    {U'ε', {147}},
    {U'↓', {148}},
    {U'#', {149}},  // Icelandic
    {U'\"', {150}}, // Russian

    {U'↑', {151}},

    // Basque
    {U'\u033a', {152}},
    {U'\u033b', {153}},

    // Luxembourgish
    {U'g', {154}},
    {U'ʦ', {155}},
    {U'X', {156}},

    // Czech
    {U'\u031d', {157}},
    {U'\u030a', {158}},
};

void phonemes_to_ids(const std::vector<Phoneme> &phonemes, PhonemeIdConfig &config,
                    std::vector<PhonemeId> &phonemeIds,
                    std::map<Phoneme, std::size_t> &missingPhonemes) {
  
  auto phonemeIdMap = std::make_shared<PhonemeIdMap>(DEFAULT_PHONEME_ID_MAP);
  if (config.phonemeIdMap) {
    phonemeIdMap = config.phonemeIdMap;
  }

  // Beginning of sentence symbol (^)
  if (config.addBos) {
    auto const bosIds = &(phonemeIdMap->at(config.bos));
    phonemeIds.insert(phonemeIds.end(), bosIds->begin(), bosIds->end());

    if (config.interspersePad) {
      // Pad after bos (_)
      auto const padIds = &(phonemeIdMap->at(config.pad));
      phonemeIds.insert(phonemeIds.end(), padIds->begin(), padIds->end());
    }
  }

  if (config.interspersePad) {
    // Add ids for each phoneme *with* padding
    auto const padIds = &(phonemeIdMap->at(config.pad));

    for (auto const phoneme : phonemes) {
      if (phonemeIdMap->count(phoneme) < 1) {
        // Phoneme is missing from id map
        if (missingPhonemes.count(phoneme) < 1) {
          missingPhonemes[phoneme] = 1;
        } else {
          missingPhonemes[phoneme] += 1;
        }

        continue;
      }

      auto const mappedIds = &(phonemeIdMap->at(phoneme));
      phonemeIds.insert(phonemeIds.end(), mappedIds->begin(), mappedIds->end());

      // pad (_)
      phonemeIds.insert(phonemeIds.end(), padIds->begin(), padIds->end());
    }
  } else {
    // Add ids for each phoneme *without* padding
    for (auto const phoneme : phonemes) {
      if (phonemeIdMap->count(phoneme) < 1) {
        // Phoneme is missing from id map
        if (missingPhonemes.count(phoneme) < 1) {
          missingPhonemes[phoneme] = 1;
        } else {
          missingPhonemes[phoneme] += 1;
        }

        continue;
      }

      auto const mappedIds = &(phonemeIdMap->at(phoneme));
      phonemeIds.insert(phonemeIds.end(), mappedIds->begin(), mappedIds->end());
    }
  }

  // End of sentence symbol ($)
  if (config.addEos) {
    auto const eosIds = &(phonemeIdMap->at(config.eos));
    phonemeIds.insert(phonemeIds.end(), eosIds->begin(), eosIds->end());
  }
}

}  // namespace piper
}  // namespace sherpa_onnx