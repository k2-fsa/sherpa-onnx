// sherpa-onnx/csrc/phrase-matcher.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_PHRASE_MATCHER_H_
#define SHERPA_ONNX_CSRC_PHRASE_MATCHER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace sherpa_onnx {

class PhraseMatcher {
 public:
  PhraseMatcher(const std::unordered_set<std::string>
                    *lexicon,  // Not owned by this instance. The passed lexicon
                               // should live longer than this instance
                const std::vector<std::string> &words, bool debug = false,
                int32_t max_search_len = 10);
  ~PhraseMatcher();

  std::vector<std::string>::const_iterator begin() const;
  std::vector<std::string>::const_iterator end() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PHRASE_MATCHER_H_
