// sherpa-onnx/csrc/piper-phonemize-lexicon.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"

#include <map>
#include <mutex>  // NOLINT

#include "espeak-ng/speak_lib.h"
#include "phoneme_ids.hpp"
#include "phonemize.hpp"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void InitEspeak(const std::string &data_dir) {
  static std::once_flag init_flag;
  std::call_once(init_flag, [data_dir]() {
    int32_t result =
        espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir.c_str(), 0);
    if (result != 22050) {
      SHERPA_ONNX_LOGE(
          "Failed to initialize espeak-ng with data dir: %s. Return code is: "
          "%d",
          data_dir.c_str(), result);
      exit(-1);
    }
  });
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(const std::string &data_dir)
    : data_dir_(data_dir) {
  InitEspeak(data_dir_);
}

std::vector<int64_t> PiperPhonemizeLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string &voice /*= ""*/) const {
  piper::eSpeakPhonemeConfig config;

  // ./bin/espeak-ng-bin --path  ./install/share/espeak-ng-data/ --voices
  // to list available voices
  config.voice = voice;  // e.g., voice is en-us

  std::vector<std::vector<piper::Phoneme>> phonemes;
  piper::phonemize_eSpeak(text, config, phonemes);

  std::vector<piper::PhonemeId> phoneme_ids;
  std::map<piper::Phoneme, std::size_t> missing_phonemes;

  std::vector<int64_t> ans;
  piper::PhonemeIdConfig id_config;
  for (const auto &p : phonemes) {
    phoneme_ids.clear();
    missing_phonemes.clear();
    phonemes_to_ids(p, id_config, phoneme_ids, missing_phonemes);
    ans.insert(ans.end(), phoneme_ids.begin(), phoneme_ids.end());
  }

  return ans;
}

}  // namespace sherpa_onnx
