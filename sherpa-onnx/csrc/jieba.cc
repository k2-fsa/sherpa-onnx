// sherpa-onnx/csrc/jieba.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/jieba.h"

#include "sherpa-onnx/csrc/file-utils.h"

namespace sherpa_onnx {

std::unique_ptr<cppjieba::Jieba> InitJieba(const std::string &dict_dir) {
  if (dict_dir.empty()) {
    return {};
  }

  std::string dict = dict_dir + "/jieba.dict.utf8";
  std::string hmm = dict_dir + "/hmm_model.utf8";
  std::string user_dict = dict_dir + "/user.dict.utf8";
  std::string idf = dict_dir + "/idf.utf8";
  std::string stop_word = dict_dir + "/stop_words.utf8";

  AssertFileExists(dict);
  AssertFileExists(hmm);
  AssertFileExists(user_dict);
  AssertFileExists(idf);
  AssertFileExists(stop_word);

  return std::make_unique<cppjieba::Jieba>(dict, hmm, user_dict, idf,
                                           stop_word);
}

}  // namespace sherpa_onnx
