// sherpa-onnx/csrc/sherpa-onnx-offline.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <iostream>

#include "sherpa-onnx/csrc/offline-transducer-model.h"

int main(int32_t argc, char *argv[]) {
  sherpa_onnx::OfflineTransducerModelConfig config;
  auto model = sherpa_onnx::OfflineTransducerModel::Create(config);
  std::cout << "model: " << model << "\n";

  return 0;
}
