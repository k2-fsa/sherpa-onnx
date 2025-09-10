// sherpa-onnx/csrc/piper-phoneme-ids.h
//
// Copyright (c)  2025  Xiaomi Corporation
// Adapted from Piper TTS phoneme_ids implementation

#ifndef SHERPA_ONNX_CSRC_PIPER_PHONEME_IDS_H_
#define SHERPA_ONNX_CSRC_PIPER_PHONEME_IDS_H_

#include <map>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/piper-phonemize.h"

namespace sherpa_onnx {
namespace piper {

typedef int64_t PhonemeId;
typedef std::map<Phoneme, std::vector<PhonemeId>> PhonemeIdMap;

struct PhonemeIdConfig {
  Phoneme pad = U'_';
  Phoneme bos = U'^';
  Phoneme eos = U'$';

  // Every other phoneme id is pad
  bool interspersePad = true;

  // Add beginning of sentence (bos) symbol at start
  bool addBos = true;

  // Add end of sentence (eos) symbol at end
  bool addEos = true;

  // Map from phonemes to phoneme id(s).
  // Not set means to use DEFAULT_PHONEME_ID_MAP.
  std::shared_ptr<PhonemeIdMap> phonemeIdMap;
};

void phonemes_to_ids(const std::vector<Phoneme> &phonemes, PhonemeIdConfig &config,
                    std::vector<PhonemeId> &phonemeIds,
                    std::map<Phoneme, std::size_t> &missingPhonemes);

}  // namespace piper
}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PIPER_PHONEME_IDS_H_