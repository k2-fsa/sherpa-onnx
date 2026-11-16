// wasm/forced-alignment/sherpa-onnx-wasm-main-fa.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include <stdio.h>

#include <algorithm>

#include "sherpa-onnx/c-api/c-api.h"

// see also
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html

extern "C" {

static_assert(sizeof(SherpaOnnxForcedAlignmentOptions) == 5 * 4, "");
static_assert(sizeof(SherpaOnnxForcedAlignerConfig) == 2 * 4 + 5 * 4 + 4, "");
static_assert(sizeof(SherpaOnnxForcedAlignmentInterval) == 3 * 4, "");
static_assert(sizeof(SherpaOnnxForcedAlignmentTier) == 3 * 4, "");
static_assert(sizeof(SherpaOnnxForcedAlignmentResult) == 4 * 4, "");

void MyPrint(SherpaOnnxForcedAlignerConfig *config) {
  fprintf(stdout, "----------forced aligner config----------\n");
  fprintf(stdout, "model_dir: %s\n", config->model_dir);
  fprintf(stdout, "lexicon: %s\n", config->lexicon);
  fprintf(stdout, "debug: %d\n", config->debug);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
