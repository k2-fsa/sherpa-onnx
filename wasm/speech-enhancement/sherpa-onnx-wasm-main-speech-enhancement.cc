// wasm/sherpa-onnx-wasm-main-speech-enhancement.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include <stdio.h>

#include <algorithm>
#include <memory>

#include "sherpa-onnx/c-api/c-api.h"

// see also
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html

extern "C" {

static_assert(sizeof(SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig) == 1 * 4,
              "");
static_assert(sizeof(SherpaOnnxOfflineSpeechDenoiserModelConfig) ==
                  sizeof(SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig) +
                      3 * 4,
              "");
static_assert(sizeof(SherpaOnnxOfflineSpeechDenoiserConfig) ==
                  sizeof(SherpaOnnxOfflineSpeechDenoiserModelConfig),
              "");

void MyPrint(SherpaOnnxOfflineSpeechDenoiserConfig *config) {
  auto model = &config->model;
  auto gtcrn = &model->gtcrn;
  fprintf(stdout, "----------offline speech denoiser model config----------\n");
  fprintf(stdout, "gtcrn: %s\n", gtcrn->model);
  fprintf(stdout, "num threads: %d\n", model->num_threads);
  fprintf(stdout, "debug: %d\n", model->debug);
  fprintf(stdout, "provider: %s\n", model->provider);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
