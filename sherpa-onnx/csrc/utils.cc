// sherpa-onnx/csrc/utils.cc
//
// Copyright      2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/utils.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/utf-utils.h"

namespace sherpa_onnx {

bool EncodeWithBpe(const std::string word,
                   const sentencepiece::SentencePieceProcessor &bpe_processor,
                   std::vector<std::string> *syms) {
  syms->clear();
  std::vector<std::string> bpes;
  if (bpe_processor.status().ok()) {
    if (bpe_processor.Encode(word, &bpes).ok()) {
      for (auto bpe : bpes) {
        if (bpe.size() >= 3) {
          // For BPE-based models, we replace ▁ with a space
          // Unicode 9601, hex 0x2581, utf8 0xe29681
          const uint8_t *p = reinterpret_cast<const uint8_t *>(bpe.c_str());
          if (p[0] == 0xe2 && p[1] == 0x96 && p[2] == 0x81) {
            bpe = bpe.replace(0, 3, " ");
          }
        }
        syms->push_back(bpe);
      }
    } else {
      SHERPA_ONNX_LOGE("SentencePiece encode error for hotword %s. ",
                       word.c_str());
      return false;
    }
  } else {
    SHERPA_ONNX_LOGE("SentencePiece processor error : %s.",
                     bpe_processor.status().ToString().c_str());
    return false;
  }
  return true;
}

bool EncodeHotwords(std::istream &is, const std::string &tokens_type,
                    const SymbolTable &symbol_table,
                    const sentencepiece::SentencePieceProcessor &bpe_processor,
                    std::vector<std::vector<int32_t>> *hotwords) {
  hotwords->clear();
  std::vector<int32_t> tmp;
  std::string line;
  std::string word;

  while (std::getline(is, line)) {
    std::istringstream iss(line);
    std::vector<std::string> syms;
    while (iss >> word) {
      if (tokens_type == "cjkchar") {
        syms.push_back(word);
      } else if (tokens_type == "bpe") {
        std::vector<std::string> bpes;
        if (!EncodeWithBpe(word, bpe_processor, &bpes)) {
          return false;
        }
        syms.insert(syms.end(), bpes.begin(), bpes.end());
      } else {
        SHERPA_ONNX_CHECK_EQ(tokens_type, "cjkchar+bpe");
        std::vector<int32_t> codes;
        if (StringToUnicodePoints(word, &codes)) {
          if (IsCJK(codes[0])) {
            syms.push_back(word);
          } else {
            std::vector<std::string> bpes;
            if (!EncodeWithBpe(word, bpe_processor, &bpes)) {
              return false;
            }
            syms.insert(syms.end(), bpes.begin(), bpes.end());
          }
        } else {
          SHERPA_ONNX_LOGE("Invalid utf8 string for hotword %s at line: %s. ",
                           word.c_str(), line.c_str());
          return false;
        }
      }
    }
    for (auto sym : syms) {
      if (symbol_table.contains(sym)) {
        int32_t number = symbol_table[sym];
        tmp.push_back(number);
      } else {
        SHERPA_ONNX_LOGE(
            "Cannot find ID for hotword %s at line: %s. (Hint: words on "
            "the "
            "same line are separated by spaces)",
            sym.c_str(), line.c_str());
        return false;
      }
    }
    hotwords->push_back(std::move(tmp));
  }
  return true;
}

}  // namespace sherpa_onnx
