// sherpa-onnx/csrc/sentence-piece-tokenizer.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SENTENCE_PIECE_TOKENIZER_H_
#define SHERPA_ONNX_CSRC_SENTENCE_PIECE_TOKENIZER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_onnx {

class SentencePieceTokenizer {
 public:
  SentencePieceTokenizer(const std::string &vocab_json,
                         const std::string &token_scores_json);
  ~SentencePieceTokenizer();

  std::vector<int32_t> EncodeIds(const std::string &text) const;
  std::vector<std::string> EncodeTokens(const std::string &text) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SENTENCE_PIECE_TOKENIZER_H_
