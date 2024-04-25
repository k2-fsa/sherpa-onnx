// wasm/sherpa-onnx-wasm-main-tts.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <stdio.h>

#include <algorithm>
#include <memory>

#include "sherpa-onnx/c-api/c-api.h"

// see also
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html

extern "C" {

static_assert(sizeof(SherpaOnnxOfflineTtsVitsModelConfig) == 8 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTtsModelConfig) ==
                  sizeof(SherpaOnnxOfflineTtsVitsModelConfig) + 3 * 4,
              "");
static_assert(sizeof(SherpaOnnxOfflineTtsConfig) ==
                  sizeof(SherpaOnnxOfflineTtsModelConfig) + 3 * 4,
              "");

void MyPrint(SherpaOnnxOfflineTtsConfig *tts_config) {
  auto tts_model_config = &tts_config->model;
  auto vits_model_config = &tts_model_config->vits;
  fprintf(stdout, "----------vits model config----------\n");
  fprintf(stdout, "model: %s\n", vits_model_config->model);
  fprintf(stdout, "lexicon: %s\n", vits_model_config->lexicon);
  fprintf(stdout, "tokens: %s\n", vits_model_config->tokens);
  fprintf(stdout, "data_dir: %s\n", vits_model_config->data_dir);
  fprintf(stdout, "noise scale: %.3f\n", vits_model_config->noise_scale);
  fprintf(stdout, "noise scale w: %.3f\n", vits_model_config->noise_scale_w);
  fprintf(stdout, "length scale: %.3f\n", vits_model_config->length_scale);
  fprintf(stdout, "dict_dir: %s\n", vits_model_config->dict_dir);

  fprintf(stdout, "----------tts model config----------\n");
  fprintf(stdout, "num threads: %d\n", tts_model_config->num_threads);
  fprintf(stdout, "debug: %d\n", tts_model_config->debug);
  fprintf(stdout, "provider: %s\n", tts_model_config->provider);

  fprintf(stdout, "----------tts config----------\n");
  fprintf(stdout, "rule_fsts: %s\n", tts_config->rule_fsts);
  fprintf(stdout, "rule_fars: %s\n", tts_config->rule_fars);
  fprintf(stdout, "max num sentences: %d\n", tts_config->max_num_sentences);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
