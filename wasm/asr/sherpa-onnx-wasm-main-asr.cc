// wasm/sherpa-onnx-wasm-main-asr.cc
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
static_assert(sizeof(SherpaOnnxFeatureConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxOnlineRecognizerConfig) ==
                  sizeof(SherpaOnnxFeatureConfig) +
                      sizeof(SherpaOnnxOnlineModelConfig) + 8 * 4,
              "");

void MyPrint(SherpaOnnxOnlineRecognizerConfig *config) {
  auto model_config = &config->model_config;
  auto feat = &config->feat_config;
  auto transducer_model_config = &model_config->transducer;
  auto paraformer_model_config = &model_config->paraformer;
  auto ctc_model_config = &model_config->zipformer2_ctc;

  fprintf(stdout, "----------online transducer model config----------\n");
  fprintf(stdout, "encoder: %s\n", transducer_model_config->encoder);
  fprintf(stdout, "decoder: %s\n", transducer_model_config->decoder);
  fprintf(stdout, "joiner: %s\n", transducer_model_config->joiner);

  fprintf(stdout, "----------online parformer model config----------\n");
  fprintf(stdout, "encoder: %s\n", paraformer_model_config->encoder);
  fprintf(stdout, "decoder: %s\n", paraformer_model_config->decoder);

  fprintf(stdout, "----------online ctc model config----------\n");
  fprintf(stdout, "model: %s\n", ctc_model_config->model);
  fprintf(stdout, "tokens: %s\n", model_config->tokens);
  fprintf(stdout, "num_threads: %d\n", model_config->num_threads);
  fprintf(stdout, "provider: %s\n", model_config->provider);
  fprintf(stdout, "debug: %d\n", model_config->debug);
  fprintf(stdout, "model type: %s\n", model_config->model_type);

  fprintf(stdout, "----------feat config----------\n");
  fprintf(stdout, "sample rate: %d\n", feat->sample_rate);
  fprintf(stdout, "feat dim: %d\n", feat->feature_dim);

  fprintf(stdout, "----------recognizer config----------\n");
  fprintf(stdout, "decoding method: %s\n", config->decoding_method);
  fprintf(stdout, "max active paths: %d\n", config->max_active_paths);
  fprintf(stdout, "enable_endpoint: %d\n", config->enable_endpoint);
  fprintf(stdout, "rule1_min_trailing_silence: %.2f\n",
          config->rule1_min_trailing_silence);
  fprintf(stdout, "rule2_min_trailing_silence: %.2f\n",
          config->rule2_min_trailing_silence);
  fprintf(stdout, "rule3_min_utterance_length: %.2f\n",
          config->rule3_min_utterance_length);
  fprintf(stdout, "hotwords_file: %s\n", config->hotwords_file);
  fprintf(stdout, "hotwords_score: %.2f\n", config->hotwords_score);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
