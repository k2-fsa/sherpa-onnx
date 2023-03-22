// sherpa-onnx/csrc/sherpa-onnx-offline.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <iostream>

#include "sherpa-onnx/csrc/offline-transducer-model.h"

int main(int32_t argc, char *argv[]) {
  sherpa_onnx::OfflineTransducerModelConfig config;
  config.encoder_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/encoder-epoch-99-avg-1.onnx";
  config.decoder_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/decoder-epoch-99-avg-1.onnx";
  config.joiner_filename =
      "./sherpa-onnx-conformer-en-2023-03-18/joiner-epoch-99-avg-1.onnx";
  config.tokens = "./sherpa-onnx-conformer-en-2023-03-18/tokens.txt";
  config.debug = true;

  sherpa_onnx::OfflineTransducerModel model(config);

  return 0;
}
