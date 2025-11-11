// sherpa-onnx/csrc/phrase-matcher.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-onnx/csrc/phrase-matcher.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {
class PhraseMatcher::Impl {
 public:
  Impl(const std::unordered_set<std::string> *lexicon,
       const std::vector<std::string> &words, bool debug,
       int32_t max_search_len)
      : lexicon_(lexicon), max_search_len_(max_search_len), debug_(debug) {
    if (max_search_len_ < 1) {
      max_search_len_ = 1;
    }
    if (debug_) {
#if __OHOS__
      SHERPA_ONNX_LOGE("max_search_len %{public}d", max_search_len_);
#else
      SHERPA_ONNX_LOGE("max_search_len %d", max_search_len_);
#endif
    }

    Build(words);

    if (debug_) {
      std::ostringstream os;
      std::string sep;
      os << "After phrase matching: ";
      for (const auto &p : phrases_) {
        os << sep << p;
        sep = "_";
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }
  }

  auto begin() const { return phrases_.begin(); }

  auto end() const { return phrases_.end(); }

 private:
  void Build(const std::vector<std::string> &words) {
    int32_t num_words = static_cast<int32_t>(words.size());
    for (int32_t i = 0; i < num_words;) {
      int32_t start = i;

      std::string w;

      if (!IsAlphaOrPunct(words[i].front())) {
        int32_t end = std::min(i + max_search_len_ - 1, num_words - 1);

        while (end > start) {
          auto this_word = GetWord(words, start, end);
          if (IsAlphaOrPunct(this_word.back())) {
            --end;
            continue;
          }

          if (debug_) {
#if __OHOS__
            SHERPA_ONNX_LOGE("%{public}d-%{public}d: %{public}s", start, end,
                             this_word.c_str());
#else
            SHERPA_ONNX_LOGE("%d-%d: %s", start, end, this_word.c_str());
#endif
          }
          if (lexicon_->count(this_word)) {
            i = end + 1;
            w = std::move(this_word);
            if (debug_) {
#if __OHOS__
              SHERPA_ONNX_LOGE("matched %{public}d-%{public}d: %{public}s",
                               start, end, w.c_str());
#else
              SHERPA_ONNX_LOGE("matched %d-%d: %s", start, end, w.c_str());
#endif
            }
            break;
          }

          end -= 1;
        }
      }

      if (w.empty()) {
        w = words[i];

        if (debug_) {
#if __OHOS__
          SHERPA_ONNX_LOGE("single word %{public}d-%{public}d: %{public}s", i,
                           i, w.c_str());
#else
          SHERPA_ONNX_LOGE("single word %d-%d: %s", i, i, w.c_str());
#endif
        }

        i += 1;
      }

      phrases_.push_back(std::move(w));
    }
  }

 private:
  std::vector<std::string> phrases_;
  const std::unordered_set<std::string> *lexicon_;
  int32_t max_search_len_;
  bool debug_;
};

PhraseMatcher::PhraseMatcher(const std::unordered_set<std::string> *lexicon,
                             const std::vector<std::string> &words,
                             bool debug /*= false*/,
                             int32_t max_search_len /*= 10*/)
    : impl_(std::make_unique<Impl>(lexicon, words, debug, max_search_len)) {}

PhraseMatcher::~PhraseMatcher() = default;

std::vector<std::string>::const_iterator PhraseMatcher::begin() const {
  return impl_->begin();
}
std::vector<std::string>::const_iterator PhraseMatcher::end() const {
  return impl_->end();
}
}  // namespace sherpa_onnx
