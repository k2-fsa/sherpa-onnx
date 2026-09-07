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
    // Support comma-separated lexicon files
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);

    for (const auto &f : files) {
      if (debug_) {
        SHERPA_ONNX_LOGE("Loading lexicon: %s", f.c_str());
      }
      auto is = OpenInputFile(f);
      InitLexicon(is, f);
    }

    AddPunctuationMappings();
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &lexicon, bool debug) : debug_(debug) {
    // Support comma-separated lexicon files
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);

    for (const auto &f : files) {
      if (debug_) {
        SHERPA_ONNX_LOGE("Loading lexicon: %s", f.c_str());
      }
      auto buf = ReadFile(mgr, f);
      std::istringstream is(std::string(buf.data(), buf.size()));
      InitLexicon(is, f);
    }

    AddPunctuationMappings();
  }

  std::vector<int32_t> ConvertTextToTokenIds(const std::string &text) const {
    // 1. Normalize punctuation
    std::string normalized = NormalizePunctuation(text);

    if (debug_) {
      SHERPA_ONNX_LOGE("Original text: %s", text.c_str());
      SHERPA_ONNX_LOGE("After normalize punctuation: %s", normalized.c_str());
    }

    // 2. Lowercase the entire text for English word matching
    std::string lowered = ToLowerCase(normalized);

    if (debug_) {
      SHERPA_ONNX_LOGE("After lowercase: %s", lowered.c_str());
    }

    // 3. Split into UTF-8 characters
    std::vector<std::string> chars = SplitUtf8(lowered);

    if (debug_) {
      SHERPA_ONNX_LOGE("After split into UTF-8 chars: %d",
                       static_cast<int32_t>(chars.size()));
    }

    // 4. Merge consecutive ASCII letters into words for English matching
    std::vector<std::string> tokens;
    std::string english_buf;
    for (const auto &ch : chars) {
      if (ch.size() == 1 && std::isalpha(static_cast<unsigned char>(ch[0]))) {
        english_buf += ch;
      } else {
        if (!english_buf.empty()) {
          tokens.push_back(english_buf);
          english_buf.clear();
        }
        tokens.push_back(ch);
      }
    }
    if (!english_buf.empty()) {
      tokens.push_back(english_buf);
    }

    if (debug_) {
      SHERPA_ONNX_LOGE("After merging English words: %d tokens",
                       static_cast<int32_t>(tokens.size()));
    }

    // 5. Use PhraseMatcher for longest match (Chinese phrases)
    PhraseMatcher matcher(&all_words_, tokens, debug_);

    std::vector<int32_t> ids;
    for (const std::string &w : matcher) {
      auto it = word2ids_.find(w);
      if (it != word2ids_.end()) {
        ids.insert(ids.end(), it->second.begin(), it->second.end());
      } else {
        // OOV: always warn
        SHERPA_ONNX_LOGE("OOV: '%s', skipping", w.c_str());
      }
    }

    return ids;
  }

 private:
  void InitLexicon(std::istream &is, const std::string &filename) {
    // Determine if this is an English lexicon by filename
    bool is_en = (filename.find("-en") != std::string::npos);

    std::string line;
    int32_t line_num = 0;

    while (std::getline(is, line)) {
      ++line_num;

      // Skip empty lines
      if (line.empty()) continue;

      // Split on " || " separator
      auto sep_pos = line.find(" || ");
      if (sep_pos == std::string::npos) {
        if (debug_) {
          SHERPA_ONNX_LOGE("Line %d: no ' || ' separator found, skipping: %s",
                           line_num, line.c_str());
        }
        continue;
      }

      std::string word = line.substr(0, sep_pos);
      std::string ids_str = line.substr(sep_pos + 4);

      if (word.empty()) continue;

      // Parse token IDs using existing utility
      std::vector<int32_t> ids;
      if (!SplitStringToIntegers(ids_str.c_str(), " ", true, &ids)) {
        if (debug_) {
          SHERPA_ONNX_LOGE("Line %d: failed to parse token IDs, skipping: %s",
                           line_num, line.c_str());
        }
        continue;
      }

      if (ids.empty()) continue;

      std::string key = word;

      if (word2ids_.count(key)) {
        if (debug_) {
          SHERPA_ONNX_LOGE("Duplicated word '%s' at line %d, ignoring",
                           key.c_str(), line_num);
        }
        continue;
      }
      word2ids_[key] = ids;
      all_words_.insert(key);

      // If English, also store lowercase version using existing utility
      if (is_en) {
        std::string lower = ToLowerCase(word);
        if (lower != key) {
          word2ids_[lower] = ids;
          all_words_.insert(lower);
        }
      }
    }

    if (debug_) {
      SHERPA_ONNX_LOGE("Loaded lexicon from %s: %d entries", filename.c_str(),
                       static_cast<int32_t>(word2ids_.size()));
    }
  }

  void AddPunctuationMappings() {
    // Add punctuation mappings (from SentencePiece, without space marker 124)
    std::vector<std::pair<std::string, int32_t>> punct_map = {
        {"!", 9676},
        {"?", 9705},
        {".", 9688},
        {",", 9686},
        {";", 9701},
        {":", 9700},
        {"(", 9682},
        {")", 9683},
        {"[", 9707},
        {"]", 9709},
        {"-", 9687},
        {"\"", 9677},
        {"'", 9},
        {"/", 9689},
        // Full-width punctuation
        {"，", 24879},
        {"。", 9729},
        {"？", 20046},
        {"、", 20094},
        {"！", 20046},
        {"；", 20094},
        {"：", 20094},
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
