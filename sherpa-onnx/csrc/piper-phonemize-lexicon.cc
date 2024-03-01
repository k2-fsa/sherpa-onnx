// sherpa-onnx/csrc/piper-phonemize-lexicon.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"

#include <codecvt>
#include <fstream>
#include <locale>
#include <map>
#include <mutex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "espeak-ng/speak_lib.h"
#include "phoneme_ids.hpp"
#include "phonemize.hpp"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

static std::unordered_map<char32_t, int32_t> ReadTokens(std::istream &is) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::unordered_map<char32_t, int32_t> token2id;

  std::string line;

  std::string sym;
  std::u32string s;
  int32_t id;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    iss >> sym;
    if (iss.eof()) {
      id = atoi(sym.c_str());
      sym = " ";
    } else {
      iss >> id;
    }

    // eat the trailing \r\n on windows
    iss >> std::ws;
    if (!iss.eof()) {
      SHERPA_ONNX_LOGE("Error when reading tokens: %s", line.c_str());
      exit(-1);
    }

    s = conv.from_bytes(sym);
    if (s.size() != 1) {
      // for tokens.txt from coqui-ai/TTS, the last token is <BLNK>
      if (s.size() == 6 && s[0] == '<' && s[1] == 'B' && s[2] == 'L' &&
          s[3] == 'N' && s[4] == 'K' && s[5] == '>') {
        continue;
      }

      SHERPA_ONNX_LOGE("Error when reading tokens at Line %s. size: %d",
                       line.c_str(), static_cast<int32_t>(s.size()));
      exit(-1);
    }

    char32_t c = s[0];

    if (token2id.count(c)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(c));
      exit(-1);
    }

    token2id.insert({c, id});
  }

  return token2id;
}

// see the function "phonemes_to_ids" from
// https://github.com/rhasspy/piper/blob/master/notebooks/piper_inference_(ONNX).ipynb
static std::vector<int64_t> PiperPhonemesToIds(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes) {
  // see
  // https://github.com/rhasspy/piper-phonemize/blob/master/src/phoneme_ids.hpp#L17
  int32_t pad = token2id.at(U'_');
  int32_t bos = token2id.at(U'^');
  int32_t eos = token2id.at(U'$');

  std::vector<int64_t> ans;
  ans.reserve(phonemes.size());

  ans.push_back(bos);
  for (auto p : phonemes) {
    if (token2id.count(p)) {
      ans.push_back(token2id.at(p));
      ans.push_back(pad);
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }
  }
  ans.push_back(eos);

  return ans;
}

static std::vector<int64_t> CoquiPhonemesToIds(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes,
    const OfflineTtsVitsModelMetaData &meta_data) {
  // see
  // https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/text/tokenizer.py#L87
  int32_t use_eos_bos = meta_data.use_eos_bos;
  int32_t bos_id = meta_data.bos_id;
  int32_t eos_id = meta_data.eos_id;
  int32_t blank_id = meta_data.blank_id;
  int32_t add_blank = meta_data.add_blank;
  int32_t comma_id = token2id.at(',');

  std::vector<int64_t> ans;
  if (add_blank) {
    ans.reserve(phonemes.size() * 2 + 3);
  } else {
    ans.reserve(phonemes.size() + 2);
  }

  if (use_eos_bos) {
    ans.push_back(bos_id);
  }

  if (add_blank) {
    ans.push_back(blank_id);

    for (auto p : phonemes) {
      if (token2id.count(p)) {
        ans.push_back(token2id.at(p));
        ans.push_back(blank_id);
      } else {
        SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                         static_cast<uint32_t>(p));
      }
    }
  } else {
    // not adding blank
    for (auto p : phonemes) {
      if (token2id.count(p)) {
        ans.push_back(token2id.at(p));
      } else {
        SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                         static_cast<uint32_t>(p));
      }
    }
  }

  // add a comma at the end of a sentence so that we can have a longer pause.
  ans.push_back(comma_id);

  if (use_eos_bos) {
    ans.push_back(eos_id);
  }

  return ans;
}

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

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &meta_data)
    : meta_data_(meta_data) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokens(is);
  }

  InitEspeak(data_dir);
}

#if __ANDROID_API__ >= 9
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &meta_data)
    : meta_data_(meta_data) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istrstream is(buf.data(), buf.size());
    token2id_ = ReadTokens(is);
  }

  // We should copy the directory of espeak-ng-data from the asset to
  // some internal or external storage and then pass the directory to
  // data_dir.
  InitEspeak(data_dir);
}
#endif

std::vector<std::vector<int64_t>> PiperPhonemizeLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string &voice /*= ""*/) const {
  piper::eSpeakPhonemeConfig config;

  // ./bin/espeak-ng-bin --path  ./install/share/espeak-ng-data/ --voices
  // to list available voices
  config.voice = voice;  // e.g., voice is en-us

  std::vector<std::vector<piper::Phoneme>> phonemes;

  static std::mutex espeak_mutex;
  {
    std::lock_guard<std::mutex> lock(espeak_mutex);

    // keep multi threads from calling into piper::phonemize_eSpeak
    piper::phonemize_eSpeak(text, config, phonemes);
  }

  std::vector<std::vector<int64_t>> ans;

  std::vector<int64_t> phoneme_ids;

  if (meta_data_.is_piper || meta_data_.is_icefall) {
    for (const auto &p : phonemes) {
      phoneme_ids = PiperPhonemesToIds(token2id_, p);
      ans.push_back(std::move(phoneme_ids));
    }
  } else if (meta_data_.is_coqui) {
    for (const auto &p : phonemes) {
      phoneme_ids = CoquiPhonemesToIds(token2id_, p, meta_data_);
      ans.push_back(std::move(phoneme_ids));
    }

  } else {
    SHERPA_ONNX_LOGE("Unsupported model");
    exit(-1);
  }

  return ans;
}

}  // namespace sherpa_onnx
