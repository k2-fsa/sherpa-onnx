// sherpa-onnx/csrc/offline-tts-zipvoice-frontend.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <algorithm>
#include <cctype>
#include <codecvt>
#include <fstream>
#include <locale>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <strstream>
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

#include "cppinyin/csrc/cppinyin.h"
#include "espeak-ng/speak_lib.h"
#include "phoneme_ids.hpp"  // NOLINT
#include "phonemize.hpp"    // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-frontend.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void CallPhonemizeEspeak(const std::string &text,
                         piper::eSpeakPhonemeConfig &config,  // NOLINT
                         std::vector<std::vector<piper::Phoneme>> *phonemes);

static std::unordered_map<std::string, int32_t> ReadTokens(std::istream &is) {
  std::unordered_map<std::string, int32_t> token2id;

  std::string line;
  std::string sym;
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
      exit(-1);
    }

    if (token2id.count(sym)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(sym));
      exit(-1);
    }
    token2id.insert({sym, id});
  }
  return token2id;
}

static std::string MapPunctuations(
    const std::string &text,
    const std::unordered_map<std::string, std::string> &punct_map) {
  std::string result = text;
  for (const auto &kv : punct_map) {
    // Replace all occurrences of kv.first with kv.second
    size_t pos = 0;
    while ((pos = result.find(kv.first, pos)) != std::string::npos) {
      result.replace(pos, kv.first.length(), kv.second);
      pos += kv.second.length();
    }
  }
  return result;
}

static void ProcessPinyin(
    const std::string &pinyin, const cppinyin::PinyinEncoder *pinyin_encoder,
    const std::unordered_map<std::string, int32_t> &token2id,
    std::vector<int64_t> *tokens_ids, std::vector<std::string> *tokens) {
  auto initial = pinyin_encoder->ToInitial(pinyin);
  if (!initial.empty()) {
    // append '0' to fix the conflict with espeak token
    initial = initial + "0";
    if (token2id.count(initial)) {
      tokens_ids->push_back(token2id.at(initial));
      tokens->push_back(initial);
    } else {
      SHERPA_ONNX_LOGE("Skip unknown initial %s", initial.c_str());
    }
  }
  auto final_t = pinyin_encoder->ToFinal(pinyin);
  if (!final_t.empty()) {
    if (!std::isdigit(final_t.back())) {
      final_t = final_t + "5";  // use 5 for neutral tone
    }
    if (token2id.count(final_t)) {
      tokens_ids->push_back(token2id.at(final_t));
      tokens->push_back(final_t);
    } else {
      SHERPA_ONNX_LOGE("Skip unknown final %s", final_t.c_str());
    }
  }
}

static void TokenizeZh(const std::string &words,
                       const cppinyin::PinyinEncoder *pinyin_encoder,
                       const std::unordered_map<std::string, int32_t> &token2id,
                       std::vector<int64_t> *token_ids,
                       std::vector<std::string> *tokens) {
  std::vector<std::string> pinyins;
  pinyin_encoder->Encode(words, &pinyins, "number" /*tone*/, false /*partial*/);
  for (const auto &pinyin : pinyins) {
    if (pinyin_encoder->ValidPinyin(pinyin, "number" /*tone*/)) {
      ProcessPinyin(pinyin, pinyin_encoder, token2id, token_ids, tokens);
    } else {
      auto wstext = ToWideString(pinyin);
      for (auto &wc : wstext) {
        auto c = ToString(std::wstring(1, wc));
        if (token2id.count(c)) {
          token_ids->push_back(token2id.at(c));
          tokens->push_back(c);
        } else {
          SHERPA_ONNX_LOGE("Skip unknown character %s", c.c_str());
        }
      }
    }
  }
}

static void TokenizeEn(const std::string &words,
                       const std::unordered_map<std::string, int32_t> &token2id,
                       const std::string &voice,
                       std::vector<int64_t> *token_ids,
                       std::vector<std::string> *tokens) {
  piper::eSpeakPhonemeConfig config;
  // ./bin/espeak-ng-bin --path  ./install/share/espeak-ng-data/ --voices
  // to list available voices
  config.voice = voice;  // e.g., voice is en-us

  std::vector<std::vector<piper::Phoneme>> phonemes;

  CallPhonemizeEspeak(words, config, &phonemes);

  for (const auto &p : phonemes) {
    for (const auto &ph : p) {
      auto token = Utf32ToUtf8(std::u32string(1, ph));
      if (token2id.count(token)) {
        token_ids->push_back(token2id.at(token));
        tokens->push_back(token);
      } else {
        SHERPA_ONNX_LOGE("Skip unknown phoneme %s", token.c_str());
      }
    }
  }
}

static void TokenizeTag(
    const std::string &words,
    const std::unordered_map<std::string, int32_t> &token2id,
    std::vector<int64_t> *tokens_ids, std::vector<std::string> *tokens) {
  // in zipvoice tags are all in upper case
  std::string tag = ToUpperAscii(words);
  if (token2id.count(tag)) {
    tokens_ids->push_back(token2id.at(tag));
    tokens->push_back(tag);
  } else {
    SHERPA_ONNX_LOGE("Skip unknown tag %s", tag.c_str());
  }
}

static void TokenizePinyin(
    const std::string &words, const cppinyin::PinyinEncoder *pinyin_encoder,
    const std::unordered_map<std::string, int32_t> &token2id,
    std::vector<int64_t> *tokens_ids, std::vector<std::string> *tokens) {
  // words are in the form of <ha3>, <ha4>
  std::string pinyin = words.substr(1, words.size() - 2);
  if (!pinyin.empty()) {
    if (pinyin[pinyin.size() - 1] == '5') {
      pinyin = pinyin.substr(0, pinyin.size() - 1);  // remove the tone
    }
    if (pinyin_encoder->ValidPinyin(pinyin, "number" /*tone*/)) {
      ProcessPinyin(pinyin, pinyin_encoder, token2id, tokens_ids, tokens);
    } else {
      SHERPA_ONNX_LOGE("Invalid pinyin %s", pinyin.c_str());
    }
  }
}

OfflineTtsZipvoiceFrontend::OfflineTtsZipvoiceFrontend(
    const std::string &tokens, const std::string &data_dir,
    const std::string &pinyin_dict,
    const OfflineTtsZipvoiceModelMetaData &meta_data, bool debug /*= false*/)
    : debug_(debug), meta_data_(meta_data) {
  std::ifstream is(tokens);
  token2id_ = ReadTokens(is);
  if (meta_data_.use_pinyin) {
    pinyin_encoder_ = std::make_unique<cppinyin::PinyinEncoder>(pinyin_dict);
  } else {
    pinyin_encoder_ = nullptr;
  }
  if (meta_data_.use_espeak) {
    // We should copy the directory of espeak-ng-data from the asset to
    // some internal or external storage and then pass the directory to
    // data_dir.
    InitEspeak(data_dir);
  }
}

template <typename Manager>
OfflineTtsZipvoiceFrontend::OfflineTtsZipvoiceFrontend(
    Manager *mgr, const std::string &tokens, const std::string &data_dir,
    const std::string &pinyin_dict,
    const OfflineTtsZipvoiceModelMetaData &meta_data, bool debug)
    : debug_(debug), meta_data_(meta_data) {
  auto buf = ReadFile(mgr, tokens);
  std::istrstream is(buf.data(), buf.size());
  token2id_ = ReadTokens(is);
  if (meta_data_.use_pinyin) {
    auto buf = ReadFile(mgr, pinyin_dict);
    std::istringstream iss(std::string(buf.begin(), buf.end()));
    pinyin_encoder_ = std::make_unique<cppinyin::PinyinEncoder>(iss);
  } else {
    pinyin_encoder_ = nullptr;
  }
  if (meta_data_.use_espeak) {
    // We should copy the directory of espeak-ng-data from the asset to
    // some internal or external storage and then pass the directory to
    // data_dir.
    InitEspeak(data_dir);
  }
}

std::vector<TokenIDs> OfflineTtsZipvoiceFrontend::ConvertTextToTokenIds(
    const std::string &_text, const std::string &voice) const {
  std::string text = _text;
  if (meta_data_.use_espeak) {
    text = ToLowerAscii(_text);
  }

  text = MapPunctuations(text, punct_map_);

  auto wstext = ToWideString(text);

  std::vector<std::string> parts;
  // Match <...>, [...], or single character
  std::wregex part_pattern(LR"([<\[].*?[>\]]|.)");
  auto words_begin =
      std::wsregex_iterator(wstext.begin(), wstext.end(), part_pattern);
  auto words_end = std::wsregex_iterator();
  for (std::wsregex_iterator i = words_begin; i != words_end; ++i) {
    parts.push_back(ToString(i->str()));
  }

  // types are en, zh, tag, pinyin, other
  // tag is [...]
  // pinyin is <...>
  // other is any other text that does not match the above, normally numbers and
  // punctuations
  std::vector<std::string> types;
  for (auto &word : parts) {
    if (word.size() == 1 && std::isalpha(word[0])) {
      // single character, e.g., 'a', 'b', 'c'
      types.push_back("en");
    } else if (word.size() > 1 && word[0] == '<' && word.back() == '>') {
      // e.g., <ha3>, <ha4>
      types.push_back("pinyin");
    } else if (word.size() > 1 && word[0] == '[' && word.back() == ']') {
      types.push_back("tag");
    } else if (ContainsCJK(word)) {  // word contains one CJK characters
      types.push_back("zh");
    } else {
      types.push_back("other");
    }
  }

  std::vector<std::pair<std::string, std::string>> parts_with_types;
  std::ostringstream oss;
  std::string t_lang;
  oss.str("");
  std::ostringstream debug_oss;
  if (debug_) {
    debug_oss << "Text : " << _text << ", Parts with types: \n";
  }
  for (int32_t i = 0; i < types.size(); ++i) {
    if (i == 0) {
      oss << parts[i];
      t_lang = types[i];
    } else {
      if (t_lang == "other" && (types[i] != "tag" && types[i] != "pinyin")) {
        // combine into current type if the previous part is "other"
        // do not combine with "tag" or "pinyin"
        oss << parts[i];
        t_lang = types[i];
      } else {
        if ((t_lang == types[i] || types[i] == "other") && t_lang != "pinyin" &&
            t_lang != "tag") {
          // same language or other, continue
          // do not combine other into "pinyin" or "tag"
          oss << parts[i];
        } else {
          // different language, start a new sentence
          std::string part = oss.str();
          oss.str("");
          parts_with_types.emplace_back(part, t_lang);
          if (debug_) {
            debug_oss << "(" << part << ", " << t_lang << "),";
          }
          oss << parts[i];
          t_lang = types[i];
        }
      }
    }
  }

  std::string part = oss.str();
  oss.str("");
  parts_with_types.emplace_back(part, t_lang);
  if (debug_) {
    debug_oss << "(" << part << ", " << t_lang << ")\n";
    SHERPA_ONNX_LOGE("%s", debug_oss.str().c_str());
    debug_oss.str("");
  }

  std::vector<int64_t> token_ids;
  std::vector<std::string> tokens;  // for debugging
  for (const auto &pt : parts_with_types) {
    if (pt.second == "zh") {
      TokenizeZh(pt.first, pinyin_encoder_.get(), token2id_, &token_ids,
                 &tokens);
    } else if (pt.second == "en") {
      TokenizeEn(pt.first, token2id_, "en-us", &token_ids, &tokens);
    } else if (pt.second == "pinyin") {
      TokenizePinyin(pt.first, pinyin_encoder_.get(), token2id_, &token_ids,
                     &tokens);
    } else if (pt.second == "tag") {
      TokenizeTag(pt.first, token2id_, &token_ids, &tokens);
    } else {
      SHERPA_ONNX_LOGE("Unexpected type: %s", pt.second.c_str());
      exit(-1);
    }
  }
  if (debug_) {
    debug_oss << "Tokens and IDs: \n";
    for (int32_t i = 0; i < tokens.size(); i++) {
      debug_oss << "(" << tokens[i] << ", " << token_ids[i] << "),";
    }
    debug_oss << "\n";
    SHERPA_ONNX_LOGE("%s", debug_oss.str().c_str());
  }

  std::vector<TokenIDs> ans;
  ans.push_back(TokenIDs(std::move(token_ids)));
  return ans;
}

#if __ANDROID_API__ >= 9
template OfflineTtsZipvoiceFrontend::OfflineTtsZipvoiceFrontend(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const std::string &pinyin_dict,
    const OfflineTtsZipvoiceModelMetaData &meta_data, bool debug = false);

#endif

#if __OHOS__
template OfflineTtsZipvoiceFrontend::OfflineTtsZipvoiceFrontend(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir, const std::string &pinyin_dict,
    const OfflineTtsZipvoiceModelMetaData &meta_data, bool debug = false);

#endif

}  // namespace sherpa_onnx
