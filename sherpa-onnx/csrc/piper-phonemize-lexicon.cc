// sherpa-onnx/csrc/piper-phonemize-lexicon.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"

#include <fstream>
#include <locale>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {
// eSpeak-based phonemization has been removed from this distribution of
// sherpa-onnx (licensing constraints). PiperPhonemizeLexicon is retained so
// that model implementations and bindings continue to compile and link, but
// it cannot convert text into phoneme token IDs. The VITS models routed
// through it -- piper/coqui/icefall/inflect -- plus Kokoro v0.19 and Kitten
// require that engine for every word, so constructing the frontend fails
// loudly with an actionable message instead of silently producing empty
// audio. Models whose frontends are purely lexicon based (e.g., Matcha's
// zh-en model, Kokoro >= v1.x passed via --kokoro-lexicon) do not use this
// class and keep working; they only lose the out-of-lexicon fallback, which
// now drops affected words with a one-time warning.
[[noreturn]] void ExitPiperPhonemizeLexiconUnsupported() {
  SHERPA_ONNX_LOGE(
      "This model needs the eSpeak text-to-phoneme engine to convert words "
      "into sounds, but this build of sherpa-onnx does not include it (it was "
      "removed for licensing reasons). Affected models: Piper / Coqui / "
      "Icefall / Inflect VITS voices (for example en_US-amy-low), Kokoro "
      "v0.19, and Kitten TTS models. Use a model with a lexicon instead: "
      "Kokoro v1.0 or newer with --kokoro-lexicon set to the matching "
      "lexicon file (for example lexicon-us-en.txt), or Matcha zh-en.");
  SHERPA_ONNX_EXIT(-1);
}
}  // namespace

static std::unordered_map<char32_t, int32_t> ReadTokens(std::istream &is) {
  std::unordered_map<char32_t, int32_t> token2id;

  std::string line;

  std::string sym;
  std::u32string s;
  int32_t id = 0;
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
      SHERPA_ONNX_EXIT(-1);
    }

    s = Utf8ToUtf32(sym);
    if (s.size() != 1) {
      // for tokens.txt from coqui-ai/TTS, the last token is <BLNK>
      if (s.size() == 6 && s[0] == '<' && s[1] == 'B' && s[2] == 'L' &&
          s[3] == 'N' && s[4] == 'K' && s[5] == '>') {
        continue;
      }

      SHERPA_ONNX_LOGE("Error when reading tokens at Line %s. size: %d",
                       line.c_str(), static_cast<int32_t>(s.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    char32_t c = s[0];

    if (token2id.count(c)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(c));
      SHERPA_ONNX_EXIT(-1);
    }

    token2id.insert({c, id});
  }

  return token2id;
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string & /*data_dir*/,
    const OfflineTtsVitsModelMetaData & /*vits_meta_data*/) {
  auto is = OpenInputFile(tokens);
  token2id_ = ReadTokens(is);
  ExitPiperPhonemizeLexiconUnsupported();
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string & /*data_dir*/,
    const OfflineTtsVitsModelMetaData & /*vits_meta_data*/) {
  auto buf = ReadFile(mgr, tokens);
  std::istringstream is(std::string(buf.data(), buf.size()));
  token2id_ = ReadTokens(is);
  ExitPiperPhonemizeLexiconUnsupported();
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string & /*data_dir*/,
    const OfflineTtsMatchaModelMetaData & /*matcha_meta_data*/) {
  auto is = OpenInputFile(tokens);
  token2id_ = ReadTokens(is);
  ExitPiperPhonemizeLexiconUnsupported();
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string & /*data_dir*/,
    const OfflineTtsMatchaModelMetaData & /*matcha_meta_data*/) {
  auto buf = ReadFile(mgr, tokens);
  std::istringstream is(std::string(buf.data(), buf.size()));
  token2id_ = ReadTokens(is);
  ExitPiperPhonemizeLexiconUnsupported();
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string & /*data_dir*/,
    const OfflineTtsKokoroModelMetaData & /*kokoro_meta_data*/) {
  auto is = OpenInputFile(tokens);
  token2id_ = ReadTokens(is);
  ExitPiperPhonemizeLexiconUnsupported();
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string & /*data_dir*/,
    const OfflineTtsKokoroModelMetaData & /*kokoro_meta_data*/) {
  auto buf = ReadFile(mgr, tokens);
  std::istringstream is(std::string(buf.data(), buf.size()));
  token2id_ = ReadTokens(is);
  ExitPiperPhonemizeLexiconUnsupported();
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string & /*data_dir*/,
    const OfflineTtsKittenModelMetaData & /*kitten_meta_data*/) {
  auto is = OpenInputFile(tokens);
  token2id_ = ReadTokens(is);
  ExitPiperPhonemizeLexiconUnsupported();
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string & /*data_dir*/,
    const OfflineTtsKittenModelMetaData & /*kitten_meta_data*/) {
  auto buf = ReadFile(mgr, tokens);
  std::istringstream is(std::string(buf.data(), buf.size()));
  token2id_ = ReadTokens(is);
  ExitPiperPhonemizeLexiconUnsupported();
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIds(
    const std::string & /*text*/, const std::string & /*voice*/ /*= ""*/)
    const {
  // Unreachable in practice: the constructors already exit. Kept as a guard
  // for the pure virtual interface.
  ExitPiperPhonemizeLexiconUnsupported();
}

#if __ANDROID_API__ >= 9
template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kitten_meta_data);
#endif

#if __OHOS__
template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsVitsModelMetaData &vits_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kitten_meta_data);
#endif

}  // namespace sherpa_onnx
