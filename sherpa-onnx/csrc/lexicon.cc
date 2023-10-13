// sherpa-onnx/csrc/lexicon.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/lexicon.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

static void ToLowerCase(std::string *in_out) {
  std::transform(in_out->begin(), in_out->end(), in_out->begin(),
                 [](unsigned char c) { return std::tolower(c); });
}

// Note: We don't use SymbolTable here since tokens may contain a blank
// in the first column
static std::unordered_map<std::string, int32_t> ReadTokens(
    const std::string &tokens) {
  std::unordered_map<std::string, int32_t> token2id;

  std::ifstream is(tokens);
  std::string line;

  std::string sym;
  int32_t id;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    iss >> sym;
    if (iss.eof()) {
      id = atoi(sym.c_str());
      sym = " ";
    } else {
      iss >> id;
    }

    if (!iss.eof()) {
      SHERPA_ONNX_LOGE("Error: %s", line.c_str());
      exit(-1);
    }

#if 0
    if (token2id.count(sym)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(sym));
      exit(-1);
    }
#endif
    token2id.insert({sym, id});
  }

  return token2id;
}

static std::vector<int32_t> ConvertTokensToIds(
    const std::unordered_map<std::string, int32_t> &token2id,
    const std::vector<std::string> &tokens) {
  std::vector<int32_t> ids;
  ids.reserve(tokens.size());
  for (const auto &s : tokens) {
    if (!token2id.count(s)) {
      return {};
    }
    int32_t id = token2id.at(s);
    ids.push_back(id);
  }

  return ids;
}

Lexicon::Lexicon(const std::string &lexicon, const std::string &tokens,
                 const std::string &punctuations) {
  token2id_ = ReadTokens(tokens);
  std::ifstream is(lexicon);

  std::string word;
  std::vector<std::string> token_list;
  std::string line;
  std::string phone;

  while (std::getline(is, line)) {
    std::istringstream iss(line);

    token_list.clear();

    iss >> word;
    ToLowerCase(&word);

    if (word2ids_.count(word)) {
      SHERPA_ONNX_LOGE("Duplicated word: %s", word.c_str());
      return;
    }

    while (iss >> phone) {
      token_list.push_back(std::move(phone));
    }

    std::vector<int32_t> ids = ConvertTokensToIds(token2id_, token_list);
    if (ids.empty()) {
      continue;
    }
    word2ids_.insert({std::move(word), std::move(ids)});
  }

  // process punctuations
  std::vector<std::string> punctuation_list;
  SplitStringToVector(punctuations, " ", false, &punctuation_list);
  for (auto &s : punctuation_list) {
    punctuations_.insert(std::move(s));
  }
}

std::vector<int64_t> Lexicon::ConvertTextToTokenIds(
    const std::string &_text) const {
  std::string text(_text);
  ToLowerCase(&text);

  std::vector<std::string> words;
  SplitStringToVector(text, " ", false, &words);

  std::vector<int64_t> ans;
  for (auto w : words) {
    std::vector<int64_t> prefix;
    while (!w.empty() && punctuations_.count(std::string(1, w[0]))) {
      // if w begins with a punctuation
      prefix.push_back(token2id_.at(std::string(1, w[0])));
      w = std::string(w.begin() + 1, w.end());
    }

    std::vector<int64_t> suffix;
    while (!w.empty() && punctuations_.count(std::string(1, w.back()))) {
      suffix.push_back(token2id_.at(std::string(1, w.back())));
      w = std::string(w.begin(), w.end() - 1);
    }

    if (!word2ids_.count(w)) {
      SHERPA_ONNX_LOGE("OOV %s. Ignore it!", w.c_str());
      continue;
    }

    const auto &token_ids = word2ids_.at(w);
    ans.insert(ans.end(), prefix.begin(), prefix.end());
    ans.insert(ans.end(), token_ids.begin(), token_ids.end());
    ans.insert(ans.end(), suffix.rbegin(), suffix.rend());
  }

  return ans;
}

}  // namespace sherpa_onnx
