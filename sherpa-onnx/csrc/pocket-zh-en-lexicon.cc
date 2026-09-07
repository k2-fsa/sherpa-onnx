// sherpa-onnx/csrc/pocket-zh-en-lexicon.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/pocket-zh-en-lexicon.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
#include "sherpa-onnx/csrc/phrase-matcher.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class PocketZhEnLexicon::Impl {
 public:
  Impl(const std::string &lexicon, bool debug) : debug_(debug) {
    auto is = OpenInputFile(lexicon);
    InitLexicon(is);
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &lexicon, bool debug) : debug_(debug) {
    auto buf = ReadFile(mgr, lexicon);
    std::istringstream is(std::string(buf.data(), buf.size()));
    InitLexicon(is);
  }

  std::vector<int32_t> ConvertTextToTokenIds(const std::string &text) const {
    // 1. Normalize punctuation
    std::string normalized = NormalizePunctuation(text);

    if (debug_) {
      SHERPA_ONNX_LOGE("Original text: %s", text.c_str());
      SHERPA_ONNX_LOGE("After normalize punctuation: %s", normalized.c_str());
    }

    // 2. Split into UTF-8 characters
    std::vector<std::string> chars = SplitUtf8(normalized);

    if (debug_) {
      SHERPA_ONNX_LOGE("After split into UTF-8 chars: %d",
                       static_cast<int32_t>(chars.size()));
    }

    // 3. Use PhraseMatcher for longest match
    PhraseMatcher matcher(&all_words_, chars, debug_);

    std::vector<int32_t> ids;
    for (const std::string &w : matcher) {
      auto it = word2ids_.find(w);
      if (it != word2ids_.end()) {
        ids.insert(ids.end(), it->second.begin(), it->second.end());
      } else {
        // OOV: skip with warning
        if (debug_) {
          SHERPA_ONNX_LOGE("OOV: '%s', skipping", w.c_str());
        }
      }
    }

    return ids;
  }

 private:
  void InitLexicon(std::istream &is) {
    std::string line;
    int32_t line_num = 0;

    while (std::getline(is, line)) {
      ++line_num;

      std::istringstream iss(line);
      std::string word;
      iss >> word;

      if (word.empty()) {
        continue;
      }

      std::vector<int32_t> ids;
      int32_t id;
      while (iss >> id) {
        ids.push_back(id);
      }

      if (!ids.empty()) {
        if (word2ids_.count(word)) {
          if (debug_) {
            SHERPA_ONNX_LOGE("Duplicated word '%s' at line %d, ignoring",
                             word.c_str(), line_num);
          }
          continue;
        }
        word2ids_[word] = ids;
        all_words_.insert(word);
      }
    }

    // Add punctuation mappings (from SentencePiece, without space marker 124)
    // These are the token IDs that SentencePiece produces for punctuation
    // after removing the space marker (124) and UNK (0)
    std::vector<std::pair<std::string, int32_t>> punct_map = {
        {"!", 9676},   {"?", 9705},  {".", 9688},   {",", 9686},
        {";", 9701},   {":", 9700},  {"(", 9682},   {")", 9683},
        {"[", 9707},   {"]", 9709},  {"-", 9687},   {"\"", 9677},
        {"'", 9},      {"/", 9689},
        // Full-width punctuation that should be kept as-is
        {"，", 24879}, {"。", 9729}, {"？", 20046}, {"、", 20094},
        {"！", 20046}, {"；", 20094}, {"：", 20094},
    };

    for (const auto &[punct, id] : punct_map) {
      if (!word2ids_.count(punct)) {
        word2ids_[punct] = {id};
        all_words_.insert(punct);
      }
    }

    if (debug_) {
      SHERPA_ONNX_LOGE("Loaded %d words from lexicon (including punctuation)",
                       static_cast<int32_t>(word2ids_.size()));
    }
  }

  std::string NormalizePunctuation(const std::string &text) const {
    // Normalize punctuation following the Python demo's rules:
    // - Keep full-width: ，。？、
    // - Convert other full-width ASCII to half-width
    // - Handle special cases
    std::string result;
    result.reserve(text.size());

    for (size_t i = 0; i < text.size();) {
      unsigned char c = text[i];
      int32_t len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
      std::string ch = text.substr(i, len);

      if (len == 1) {
        // ASCII character - keep as-is
        result += ch;
      } else if (ch == "，" || ch == "。" || ch == "？" || ch == "、") {
        // Keep these full-width punctuation as-is
        result += ch;
      } else if (ch == "！") {
        // Convert full-width ! to half-width
        result += "!";
      } else if (ch == "；") {
        result += ";";
      } else if (ch == "：") {
        result += ":";
      } else if (ch == "（") {
        result += "(";
      } else if (ch == "）") {
        result += ")";
      } else if (ch == "【") {
        result += "[";
      } else if (ch == "】") {
        result += "]";
      } else if (ch == "\"" || ch == "\"" || ch == "「" || ch == "」" ||
                 ch == "『" || ch == "』") {
        result += "\"";
      } else if (ch == "'" || ch == "'") {
        result += "'";
      } else if (ch == "…") {
        result += "...";
      } else if (ch == "｡") {
        result += ".";
      } else {
        result += ch;
      }

      i += len;
    }

    return result;
  }

  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;
  std::unordered_set<std::string> all_words_;
  bool debug_;
};

PocketZhEnLexicon::PocketZhEnLexicon(const std::string &lexicon, bool debug)
    : impl_(std::make_unique<Impl>(lexicon, debug)) {}

template <typename Manager>
PocketZhEnLexicon::PocketZhEnLexicon(Manager *mgr, const std::string &lexicon,
                                     bool debug)
    : impl_(std::make_unique<Impl>(mgr, lexicon, debug)) {}

PocketZhEnLexicon::~PocketZhEnLexicon() = default;

std::vector<int32_t> PocketZhEnLexicon::ConvertTextToTokenIds(
    const std::string &text) const {
  return impl_->ConvertTextToTokenIds(text);
}

#if __ANDROID_API__ >= 9
template PocketZhEnLexicon::PocketZhEnLexicon(AAssetManager *mgr,
                                              const std::string &lexicon,
                                              bool debug);
#endif

#if __OHOS__
template PocketZhEnLexicon::PocketZhEnLexicon(NativeResourceManager *mgr,
                                              const std::string &lexicon,
                                              bool debug);
#endif

}  // namespace sherpa_onnx
