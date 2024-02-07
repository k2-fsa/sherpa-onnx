// wasm/sherpa-onnx-wasm-main.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <algorithm>
#include <memory>

#include "sherpa-onnx/c-api/c-api.h"

// see also
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html

extern "C" {

static_assert(sizeof(SherpaOnnxOfflineTtsVitsModelConfig) == 7 * 4, "");

void MyPrint(SherpaOnnxOfflineTtsVitsModelConfig *vits_model_config) {
  fprintf(stdout, "----------vits model config----------\n");
  fprintf(stdout, "model: %s\n", vits_model_config->model);
  fprintf(stdout, "lexicon: %s\n", vits_model_config->lexicon);
  fprintf(stdout, "tokens: %s\n", vits_model_config->tokens);
  fprintf(stdout, "data_dir: %s\n", vits_model_config->data_dir);
  fprintf(stdout, "noise scale: %.3f\n", vits_model_config->noise_scale);
  fprintf(stdout, "noise scale w: %.3f\n", vits_model_config->noise_scale_w);
  fprintf(stdout, "length scale: %.3f\n", vits_model_config->length_scale);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
