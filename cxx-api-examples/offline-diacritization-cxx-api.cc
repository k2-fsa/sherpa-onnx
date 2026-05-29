// cxx-api-examples/offline-diacritization-cxx-api.cc
// Copyright (c)  2026  Matias Lin

// To use the CATT (Encoder-Only) diacritization model:
// clang-format off
// wget https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip
// unzip eo_model_onnx.zip -d catt_eo_model_onnx
// rm eo_model_onnx.zip
// clang-format on

#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main() {
  using namespace sherpa_onnx::cxx;  // NOLINT

  OfflineDiacritizationConfig diacritization_config;
  diacritization_config.model.catt_encoder =
      "./catt_eo_model_onnx/encoder.onnx";
  diacritization_config.model.catt_decoder =
      "./catt_eo_model_onnx/decoder.onnx";
  diacritization_config.model.num_threads = 1;
  diacritization_config.model.debug = false;
  diacritization_config.model.provider = "cpu";

  OfflineDiacritization diacrt =
      OfflineDiacritization::Create(diacritization_config);
  if (!diacrt.Get()) {
    std::cerr
        << "Failed to create diacritization model. Please check your config\n";
    return -1;
  }

  std::string text = "اللغة العربية من أقدم اللغات السامية";
  std::string text_with_diacritics = diacrt.AddDiacritics(text);
  std::cout << "Original text: " << text << std::endl;
  std::cout << "With diacritics: " << text_with_diacritics << std::endl;

  return 0;
}
