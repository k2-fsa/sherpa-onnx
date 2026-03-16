// c-api-examples/add-punctuation-online-c-api.c
//
// Copyright (c)  zengyw

// We assume you have pre-downloaded the model files for testing
// from https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
//
// An example is given below:
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
// tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
// rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  SherpaOnnxOnlinePunctuationConfig config;
  memset(&config, 0, sizeof(config));

  // clang-format off
  config.model.cnn_bilstm = "./sherpa-onnx-online-punct-en-2024-08-06/model.int8.onnx";
  config.model.bpe_vocab = "./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab";
  // clang-format on
  config.model.num_threads = 1;
  config.model.debug = 1;
  config.model.provider = "cpu";

  const SherpaOnnxOnlinePunctuation *punct =
      SherpaOnnxCreateOnlinePunctuation(&config);
  if (!punct) {
    fprintf(stderr,
            "Failed to create OnlinePunctuation. Please check your config\n");
    return -1;
  }

  const char *texts[] = {
      "how are you i am fine thank you",
      ("The African blogosphere is rapidly expanding bringing more voices "
       "online in the form of commentaries opinions analyses rants and poetry"),
  };

  int32_t n = sizeof(texts) / sizeof(const char *);
  fprintf(stderr, "n: %d\n", n);

  fprintf(stderr, "--------------------\n");
  for (int32_t i = 0; i != n; ++i) {
    const char *text_with_punct =
        SherpaOnnxOnlinePunctuationAddPunct(punct, texts[i]);
    if (!text_with_punct) {
      fprintf(stderr, "Failed to add punctuation for: %s\n", texts[i]);
      continue;
    }

    fprintf(stderr, "Input text: %s\n", texts[i]);
    fprintf(stderr, "Output text: %s\n", text_with_punct);
    SherpaOnnxOnlinePunctuationFreeText(text_with_punct);
    fprintf(stderr, "--------------------\n");
  }

  SherpaOnnxDestroyOnlinePunctuation(punct);

  return 0;
}
