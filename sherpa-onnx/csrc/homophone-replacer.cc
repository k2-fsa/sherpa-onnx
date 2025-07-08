// sherpa-onnx/csrc/homophone-replacer.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/homophone-replacer.h"

#include <fstream>
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

#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/jieba.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void HomophoneReplacerConfig::Register(ParseOptions *po) {
  po->Register("hr-dict-dir", &dict_dir,
               "The dict directory for jieba used by HomophoneReplacer");

  po->Register("hr-lexicon", &lexicon,
               "Path to lexicon.txt used by HomophoneReplacer.");

  po->Register("hr-rule-fsts", &rule_fsts,
               "Fst files for HomophoneReplacer. If there are multiple, they "
               "are separated by a comma. E.g., a.fst,b.fst,c.fst");
}

bool HomophoneReplacerConfig::Validate() const {
  if (!dict_dir.empty()) {
    std::vector<std::string> required_files = {
        "jieba.dict.utf8", "hmm_model.utf8",  "user.dict.utf8",
        "idf.utf8",        "stop_words.utf8",
    };

    for (const auto &f : required_files) {
      if (!FileExists(dict_dir + "/" + f)) {
        SHERPA_ONNX_LOGE("'%s/%s' does not exist. Please check kokoro-dict-dir",
                         dict_dir.c_str(), f.c_str());
        return false;
      }
    }
  }

  if (!lexicon.empty() && !FileExists(lexicon)) {
    SHERPA_ONNX_LOGE("--hr-lexicon: '%s' does not exist", lexicon.c_str());
    return false;
  }

  if (!rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fsts, ",", false, &files);

    if (files.size() > 1) {
      SHERPA_ONNX_LOGE("Only 1 file is supported now.");
      SHERPA_ONNX_EXIT(-1);
    }

    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule fst '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  return true;
}

std::string HomophoneReplacerConfig::ToString() const {
  std::ostringstream os;

  os << "HomophoneReplacerConfig(";
  os << "dict_dir=\"" << dict_dir << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "rule_fsts=\"" << rule_fsts << "\")";

  return os.str();
}

class HomophoneReplacer::Impl {
 public:
  explicit Impl(const HomophoneReplacerConfig &config) : config_(config) {
    jieba_ = InitJieba(config.dict_dir);

    {
      std::ifstream is(config.lexicon);
      InitLexicon(is);
    }

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      replacer_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.debug) {
          SHERPA_ONNX_LOGE("hr rule fst: %s", f.c_str());
        }
        replacer_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
      }
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const HomophoneReplacerConfig &config) : config_(config) {
    jieba_ = InitJieba(config.dict_dir);
    {
      auto buf = ReadFile(mgr, config.lexicon);

      std::istrstream is(buf.data(), buf.size());
      InitLexicon(is);
    }

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      replacer_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.debug) {
          SHERPA_ONNX_LOGE("hr rule fst: %s", f.c_str());
        }
        auto buf = ReadFile(mgr, f);
        std::istrstream is(buf.data(), buf.size());
        replacer_list_.push_back(
            std::make_unique<kaldifst::TextNormalizer>(is));
      }
    }
  }

  std::string Apply(const std::string &text) const {
    std::string ans;

    if (text.empty()) {
      return ans;
    }

    bool is_hmm = true;

    std::vector<std::string> words;
    jieba_->Cut(text, words, is_hmm);
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Input text: '%s'", text.c_str());
      std::ostringstream os;
      os << "After jieba: ";
      std::string sep;
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }

    // convert words to pronunciations
    std::vector<std::string> current_words;
    std::vector<std::string> current_pronunciations;

    for (const auto &w : words) {
      if (w.size() < 3 ||
          reinterpret_cast<const uint8_t *>(w.data())[0] < 128) {
        if (!current_words.empty()) {
          ans += ApplyImpl(current_words, current_pronunciations);
          current_words.clear();
          current_pronunciations.clear();
        }
        ans += w;
        continue;
      }

      auto p = ConvertWordToPronunciation(w);
      if (config_.debug) {
        SHERPA_ONNX_LOGE("%s %s", w.c_str(), p.c_str());
      }

      current_words.push_back(w);
      current_pronunciations.push_back(std::move(p));
    }

    if (!current_words.empty()) {
      ans += ApplyImpl(current_words, current_pronunciations);
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Output text: '%s'", ans.c_str());
    }

    return ans;
  }

 private:
  std::string ApplyImpl(const std::vector<std::string> &words,
                        const std::vector<std::string> &pronunciations) const {
    std::string ans;
    for (const auto &r : replacer_list_) {
      ans = r->Normalize(words, pronunciations);
      // TODO(fangjun): We support only 1 rule fst at present.
      break;
    }
    return ans;
  }
  std::string ConvertWordToPronunciation(const std::string &word) const {
    if (word2pron_.count(word)) {
      return word2pron_.at(word);
    }

    if (word.size() <= 3) {
      // not a Chinese character
      return word;
    }

    std::vector<std::string> words = SplitUtf8(word);
    std::string ans;
    for (const auto &w : words) {
      if (word2pron_.count(w)) {
        ans.append(word2pron_.at(w));
      } else {
        ans.append(w);
      }
    }

    return ans;
  }

  void InitLexicon(std::istream &is) {
    std::string word;
    std::string pron;
    std::string p;

    std::string line;
    int32_t line_num = 0;
    int32_t num_warn = 0;
    while (std::getline(is, line)) {
      ++line_num;
      std::istringstream iss(line);

      pron.clear();
      iss >> word;
      ToLowerCase(&word);

      if (word2pron_.count(word)) {
        num_warn += 1;
        if (num_warn < 10) {
          SHERPA_ONNX_LOGE("Duplicated word: %s at line %d:%s. Ignore it.",
                           word.c_str(), line_num, line.c_str());
        }
        continue;
      }

      while (iss >> p) {
        if (p.back() > '4') {
          p.push_back('1');
        }
        pron.append(std::move(p));
      }

      if (pron.empty()) {
        SHERPA_ONNX_LOGE(
            "Empty pronunciation for word '%s' at line %d:%s. Ignore it.",
            word.c_str(), line_num, line.c_str());
        continue;
      }

      word2pron_.insert({std::move(word), std::move(pron)});
    }
  }

 private:
  HomophoneReplacerConfig config_;
  std::unique_ptr<cppjieba::Jieba> jieba_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> replacer_list_;
  std::unordered_map<std::string, std::string> word2pron_;
};

HomophoneReplacer::HomophoneReplacer(const HomophoneReplacerConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
HomophoneReplacer::HomophoneReplacer(Manager *mgr,
                                     const HomophoneReplacerConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

HomophoneReplacer::~HomophoneReplacer() = default;

std::string HomophoneReplacer::Apply(const std::string &text) const {
  return RemoveInvalidUtf8Sequences(impl_->Apply(text));
}

#if __ANDROID_API__ >= 9
template HomophoneReplacer::HomophoneReplacer(
    AAssetManager *mgr, const HomophoneReplacerConfig &config);
#endif

#if __OHOS__
template HomophoneReplacer::HomophoneReplacer(
    NativeResourceManager *mgr, const HomophoneReplacerConfig &config);
#endif

}  // namespace sherpa_onnx
