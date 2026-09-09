// sherpa-onnx/csrc/pocket-zh-en-lexicon.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_POCKET_ZH_EN_LEXICON_H_
#define SHERPA_ONNX_CSRC_POCKET_ZH_EN_LEXICON_H_

#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

class PocketZhEnLexicon {
 public:
  PocketZhEnLexicon(const std::string &lexicon, bool debug);

  template <typename Manager>
  PocketZhEnLexicon(Manager *mgr, const std::string &lexicon, bool debug);

  ~PocketZhEnLexicon();

  // Convert text to token IDs using longest match
  std::vector<int32_t> ConvertTextToTokenIds(const std::string &text) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_POCKET_ZH_EN_LEXICON_H_
