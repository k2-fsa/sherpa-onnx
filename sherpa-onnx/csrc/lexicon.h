// sherpa-onnx/csrc/lexicon.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_LEXICON_H_
#define SHERPA_ONNX_CSRC_LEXICON_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sherpa_onnx {

class Lexicon {
 public:
  Lexicon(const std::string &lexicon, const std::string &tokens,
          const std::string &punctuations);

  std::vector<int64_t> ConvertTextToTokenIds(const std::string &text) const;

 private:
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;
  std::unordered_set<std::string> punctuations_;
  std::unordered_map<std::string, int32_t> token2id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_LEXICON_H_
