// sherpa-onnx/csrc/jieba.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/jieba.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

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

#if __ANDROID_API__ >= 9 || defined(__OHOS__)
  if (dict[0] != '/') {
    SHERPA_ONNX_LOGE(
        "You need to follow our examples to copy the jieba dict directory from "
        "the assets folder to an external storage directory");

    SHERPA_ONNX_LOGE(
        "Hint: Please see\n"
        "https://github.com/k2-fsa/sherpa-onnx/blob/master/android/"
        "SherpaOnnxTtsEngine/app/src/main/java/com/k2fsa/sherpa/onnx/tts/"
        "engine/TtsEngine.kt#L193\n"
        "The function copyDataDir()\n");
  }
#endif

  AssertFileExists(dict);
  AssertFileExists(hmm);
  AssertFileExists(user_dict);
  AssertFileExists(idf);
  AssertFileExists(stop_word);

  return std::make_unique<cppjieba::Jieba>(dict, hmm, user_dict, idf,
                                           stop_word);
}

}  // namespace sherpa_onnx
