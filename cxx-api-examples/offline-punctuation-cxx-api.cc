// cxx-api-examples/offline-punctuation-cxx-api.cc
// Copyright (c)  2025  Xiaomi Corporation

// To use punctuation model:
// clang-format off
// wget
// https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8..tar.bz2
// tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2 
// rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2
// clang-format on

#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  OfflinePunctuationConfig punctuation_config;
  punctuation_config.model.ct_transformer =
      "./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/"
      "model.int8.onnx";
  punctuation_config.model.num_threads = 1;
  punctuation_config.model.debug = false;
  punctuation_config.model.provider = "cpu";

  OfflinePunctuation punct = OfflinePunctuation::Create(punctuation_config);
  if (!punct.Get()) {
    std::cerr
        << "Failed to create punctuation model. Please check your config\n";
    return -1;
  }

  std::string text = "你好吗how are you Fantasitic 谢谢我很好你怎么样呢";
  std::string text_with_punct = punct.AddPunctuation(text);
  std::cout << "Original text: " << text << std::endl;
  std::cout << "With punctuation: " << text_with_punct << std::endl;

  return 0;
}
