// sherpa-onnx/csrc/tashkeel-tokenizer.h
//
// Adapted from CATT tashkeel_tokenizer.py
//
// Copyright (c)  2026  Matias Lin
#ifndef SHERPA_ONNX_CSRC_TASHKEEL_TOKENIZER_H_
#define SHERPA_ONNX_CSRC_TASHKEEL_TOKENIZER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sherpa_onnx {

class TashkeelTokenizer {
 public:
  TashkeelTokenizer();

  struct EncodeResult {
    std::vector<int64_t> input_ids_;
    std::vector<int64_t> target_ids_;
  };

  EncodeResult Encode(const std::string &text) const;
  std::string Decode(const std::vector<int64_t> &input_ids,
                     const std::vector<int64_t> &target_ids) const;

  std::string CleanText(const std::string &text) const;

  int64_t NoTashkeelId() const;   // <NT>
  int64_t SpaceLetterId() const;  // " "

 private:
  // Buckwalter transliteration
  std::string Ar2Bw(const std::string &text) const;
  std::string Bw2Ar(const std::string &text) const;

  using LetterTashkeelPairList =
      std::vector<std::pair<std::string, std::string>>;

  std::string UnifyShaddahPosition(const std::string &text) const;
  std::string DedupConsecutiveHarakat(const std::string &text) const;
  LetterTashkeelPairList SplitTashkeelFromText(
      const std::string &bw_text) const;
  std::string CombineTashkeelWithText(const LetterTashkeelPairList &text) const;
  std::vector<std::string> FilterTashkeel(
      const std::vector<std::string> &tashkeel) const;

  std::vector<std::string> letters_;
  std::vector<std::string> tashkeel_list_;
  std::vector<std::string> shaddah_last_;
  std::vector<std::string> shaddah_first_;
  std::vector<std::string> tashkeel_chars_;

  std::unordered_map<std::string, uint64_t> letters_map_;
  std::unordered_map<std::string, uint64_t> tashkeel_map_;
  std::unordered_map<std::string, std::string> inverse_tags_;
  std::unordered_map<std::string, std::string> tags_;

  int64_t no_tashkeel_id_;
  int64_t space_letter_id_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TASHKEEL_TOKENIZER_H_
