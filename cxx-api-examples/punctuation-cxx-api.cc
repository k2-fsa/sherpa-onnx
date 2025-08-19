// cxx-api-examples/fire-red-asr-cxx-api.cc
// Copyright (c)  2025  Xiaomi Corporation

//
// This file demonstrates how to use FireRedAsr AED with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
// tar xvf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
// rm sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
//
// clang-format on

#include <chrono>  // NOLINT
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  const char *kUsageMessage = R"usage(
Add punctuations to the input text.

The input text can contain both Chinese and English words.

Usage:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

./bin/sherpa-onnx-offline-punctuation \
  --ct-transformer=./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx
  "你好吗how are you Fantasitic 谢谢我很好你怎么样呢"

The output text should look like below:
)usage";
  using namespace sherpa_onnx::cxx;  // NOLINT
  OfflinePunctuationConfig punctuation_config;
  punctuation_config.model.ct_transformer = "./models/punctuation.onnx";
  punctuation_config.model.num_threads = 1;
  punctuation_config.model.debug = false;
  punctuation_config.model.provider = "cpu";
  OfflinePunctuation punct = OfflinePunctuation::Create(punctuation_config);
  std::string text = "你好吗how are you Fantasitic 谢谢我很好你怎么样呢";
  std::string text_with_punct = punct.AddPunctuation(text);
  std::cout << "Original text: " << text << std::endl;
  std::cout << "With punctuation: " << text_with_punct << std::endl;
}
