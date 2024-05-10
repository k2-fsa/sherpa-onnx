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

static bool EncodeBase(std::istream &is, const SymbolTable &symbol_table,
                       std::vector<std::vector<int32_t>> *ids,
                       std::vector<std::string> *phrases,
                       std::vector<float> *scores,
                       std::vector<float> *thresholds) {
  SHERPA_ONNX_CHECK(ids != nullptr);
  ids->clear();

  std::vector<int32_t> tmp_ids;
  std::vector<float> tmp_scores;
  std::vector<float> tmp_thresholds;
  std::vector<std::string> tmp_phrases;

  std::string line;
  std::string word;
  bool has_scores = false;
  bool has_thresholds = false;
  bool has_phrases = false;

  while (std::getline(is, line)) {
    float score = 0;
    float threshold = 0;
    std::string phrase = "";

    std::istringstream iss(line);
    while (iss >> word) {
      if (word.size() >= 3) {
        // For BPE-based models, we replace ▁ with a space
        // Unicode 9601, hex 0x2581, utf8 0xe29681
        const uint8_t *p = reinterpret_cast<const uint8_t *>(word.c_str());
        if (p[0] == 0xe2 && p[1] == 0x96 && p[2] == 0x81) {
          word = word.replace(0, 3, " ");
        }
      }
      if (symbol_table.Contains(word)) {
        int32_t id = symbol_table[word];
        tmp_ids.push_back(id);
      } else {
        switch (word[0]) {
          case ':':  // boosting score for current keyword
            score = std::stof(word.substr(1));
            has_scores = true;
            break;
          case '#':  // triggering threshold (probability) for current keyword
            threshold = std::stof(word.substr(1));
            has_thresholds = true;
            break;
          case '@':  // the original keyword string
            phrase = word.substr(1);
            has_phrases = true;
            break;
          default:
            SHERPA_ONNX_LOGE(
                "Cannot find ID for token %s at line: %s. (Hint: words on "
                "the same line are separated by spaces)",
                word.c_str(), line.c_str());
            return false;
        }
      }
    }
    ids->push_back(std::move(tmp_ids));
    tmp_scores.push_back(score);
    tmp_phrases.push_back(phrase);
    tmp_thresholds.push_back(threshold);
  }
  if (scores != nullptr) {
    if (has_scores) {
      scores->swap(tmp_scores);
    } else {
      scores->clear();
    }
  }
  if (phrases != nullptr) {
    if (has_phrases) {
      *phrases = std::move(tmp_phrases);
    } else {
      phrases->clear();
    }
  }
  if (thresholds != nullptr) {
    if (has_thresholds) {
      thresholds->swap(tmp_thresholds);
    } else {
      thresholds->clear();
    }
  }
  return true;
}

bool EncodeHotwords(std::istream &is, const SymbolTable &symbol_table,
                    std::vector<std::vector<int32_t>> *hotwords) {
  return EncodeBase(is, symbol_table, hotwords, nullptr, nullptr, nullptr);
}

bool EncodeKeywords(std::istream &is, const SymbolTable &symbol_table,
                    std::vector<std::vector<int32_t>> *keywords_id,
                    std::vector<std::string> *keywords,
                    std::vector<float> *boost_scores,
                    std::vector<float> *threshold) {
  return EncodeBase(is, symbol_table, keywords_id, keywords, boost_scores,
                    threshold);
}

}  // namespace sherpa_onnx
