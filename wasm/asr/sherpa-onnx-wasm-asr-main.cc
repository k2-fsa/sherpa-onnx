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
static_assert(sizeof(SherpaOnnxOnlineModelConfig) ==
                  sizeof(SherpaOnnxOnlineTransducerModelConfig) +
                      sizeof(SherpaOnnxOnlineParaformerModelConfig) +
                      sizeof(SherpaOnnxOnlineZipformer2CtcModelConfig) + 5 * 4,
              "");

void MyPrint(SherpaOnnxOnlineModelConfig *config) {
  auto transducer_model_config = &config->transducer;
  auto paraformer_model_config = &config->paraformer;
  auto ctc_model_config = &config->zipformer2_ctc;

  fprintf(stdout, "----------online transducer model config----------\n");
  fprintf(stdout, "encoder: %s\n", transducer_model_config->encoder);
  fprintf(stdout, "decoder: %s\n", transducer_model_config->decoder);
  fprintf(stdout, "joiner: %s\n", transducer_model_config->joiner);

  fprintf(stdout, "----------online parformer model config----------\n");
  fprintf(stdout, "encoder: %s\n", paraformer_model_config->encoder);
  fprintf(stdout, "decoder: %s\n", paraformer_model_config->decoder);

  fprintf(stdout, "----------online ctc model config----------\n");
  fprintf(stdout, "model: %s\n", ctc_model_config->model);
  fprintf(stdout, "tokens: %s\n", config->tokens);
  fprintf(stdout, "num_threads: %d\n", config->num_threads);
  fprintf(stdout, "provider: %s\n", config->provider);
  fprintf(stdout, "debug: %d\n", config->debug);
  fprintf(stdout, "model type: %s\n", config->model_type);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
