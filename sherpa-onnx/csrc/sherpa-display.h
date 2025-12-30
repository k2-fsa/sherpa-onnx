// sherpa-onnx/csrc/sherpa-display.h
//
// Copyright (c)  2025  Xiaomi Corporation
#pragma once

#include <stdlib.h>

#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sherpa_onnx {

class SherpaDisplay {
 public:
  void UpdateText(const std::string &text) { current_text_ = text; }

  void FinalizeCurrentSentence() {
    if (!current_text_.empty() &&
        (current_text_[0] != ' ' || current_text_.size() > 1)) {
      sentences_.push_back({GetCurrentDateTime(), std::move(current_text_)});
    }
  }

  void Display() const {
    if (!sentences_.empty() || !current_text_.empty()) {
      ClearScreen();
    }

    printf("=== Speech Recognition with Next-gen Kaldi ===\n");
    printf("------------------------------\n");
    if (!sentences_.empty()) {
      int32_t i = 1;
      for (const auto &p : sentences_) {
        printf("[%s] %d. %s\n", p.first.c_str(), i, p.second.c_str());
        i += 1;
      }

      printf("------------------------------\n");
    }

    if (!current_text_.empty()) {
      printf("Recognizing: %s\n", current_text_.c_str());
    }
  }

 private:
  static void ClearScreen() {
#ifdef _MSC_VER
    auto ret = system("cls");
#else
    auto ret = system("clear");
#endif
    (void)ret;
  }

  static std::string GetCurrentDateTime() {
    std::ostringstream os;
    auto t = std::time(nullptr);
    auto tm = std::localtime(&t);
    os << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
    return os.str();
  }

 private:
  std::vector<std::pair<std::string, std::string>> sentences_;
  std::string current_text_;
};

}  // namespace sherpa_onnx
