// wasm/sherpa-onnx-wasm-main-kws.cc
//
// Copyright (c)  2024  lovemefan
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
                      sizeof(SherpaOnnxOnlineZipformer2CtcModelConfig) + 9 * 4,
              "");
static_assert(sizeof(SherpaOnnxFeatureConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxKeywordSpotterConfig) ==
                  sizeof(SherpaOnnxFeatureConfig) +
                      sizeof(SherpaOnnxOnlineModelConfig) + 7 * 4,
              "");

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
