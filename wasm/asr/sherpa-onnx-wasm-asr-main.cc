// wasm/sherpa-onnx-wasm-main.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <stdio.h>

#include <algorithm>
#include <memory>

#include "sherpa-onnx/c-api/c-api.h"

// see also
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html

extern "C" {

static_assert(sizeof(SherpaOnnxOnlineTransducerModelConfig) == 3 * 4, "");
static_assert(sizeof(SherpaOnnxOnlineParaformerModelConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxOnlineZipformer2CtcModelConfig) == 1 * 4, "");

void MyPrint(SherpaOnnxOnlineTransducerModelConfig *transducer_model_config,
             SherpaOnnxOnlineParaformerModelConfig *paraformer_model_config,
             SherpaOnnxOnlineZipformer2CtcModelConfig *ctc_model_config) {
  fprintf(stdout, "----------online transducer model config----------\n");
  fprintf(stdout, "encoder: %s\n", transducer_model_config->encoder);
  fprintf(stdout, "decoder: %s\n", transducer_model_config->decoder);
  fprintf(stdout, "joiner: %s\n", transducer_model_config->joiner);

  fprintf(stdout, "----------online parformer model config----------\n");
  fprintf(stdout, "encoder: %s\n", paraformer_model_config->encoder);
  fprintf(stdout, "decoder: %s\n", paraformer_model_config->decoder);

  fprintf(stdout, "----------online ctc model config----------\n");
  fprintf(stdout, "model: %s\n", ctc_model_config->model);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
