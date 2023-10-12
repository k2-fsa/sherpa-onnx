// sherpa-onnx/csrc/utils.cc
//
// Copyright      2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/utils.h"

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

bool EncodeHotwords(std::istream &is, const SymbolTable &symbol_table,
                    std::vector<std::vector<int32_t>> *hotwords) {
  hotwords->clear();
  std::vector<int32_t> tmp;
  std::string line;
  std::string word;

  while (std::getline(is, line)) {
    std::istringstream iss(line);
    std::vector<std::string> syms;
    while (iss >> word) {
      if (word.size() >= 3) {
        // For BPE-based models, we replace ▁ with a space
        // Unicode 9601, hex 0x2581, utf8 0xe29681
        const uint8_t *p = reinterpret_cast<const uint8_t *>(word.c_str());
        if (p[0] == 0xe2 && p[1] == 0x96 && p[2] == 0x81) {
          word = word.replace(0, 3, " ");
        }
      }
      if (symbol_table.contains(word)) {
        int32_t number = symbol_table[word];
        tmp.push_back(number);
      } else {
        SHERPA_ONNX_LOGE(
            "Cannot find ID for hotword %s at line: %s. (Hint: words on "
            "the "
            "same line are separated by spaces)",
            word.c_str(), line.c_str());
        return false;
      }
    }
    hotwords->push_back(std::move(tmp));
  }
  return true;
}

}  // namespace sherpa_onnx
