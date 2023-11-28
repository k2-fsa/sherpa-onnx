// sherpa-onnx/csrc/piper-phonemize-lexicon.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_LEXICON_H_
#define SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_LEXICON_H_

#include "sherpa-onnx/csrc/lexicon.h"

namespace sherpa_onnx {

class PiperPhonemizeLexicon : public Lexicon {
 public:
  explicit PiperPhonemizeLexicon(const std::string &data_dir);

  std::vector<int64_t> ConvertTextToTokenIds(
      const std::string &text, const std::string &voice = "") const override;

 private:
  std::string voice_;
  std::string data_dir_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PIPER_PHONEMIZE_LEXICON_H_
