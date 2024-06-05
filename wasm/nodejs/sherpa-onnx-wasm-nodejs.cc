// wasm/sherpa-onnx-wasm-main-nodejs.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <stdio.h>

#include <algorithm>
#include <memory>

#include "sherpa-onnx/c-api/c-api.h"

extern "C" {

static_assert(sizeof(SherpaOnnxOfflineTransducerModelConfig) == 3 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineParaformerModelConfig) == 4, "");

static_assert(sizeof(SherpaOnnxOfflineNemoEncDecCtcModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineWhisperModelConfig) == 5 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTdnnModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineLMConfig) == 2 * 4, "");

static_assert(sizeof(SherpaOnnxOfflineModelConfig) ==
                  sizeof(SherpaOnnxOfflineTransducerModelConfig) +
                      sizeof(SherpaOnnxOfflineParaformerModelConfig) +
                      sizeof(SherpaOnnxOfflineNemoEncDecCtcModelConfig) +
                      sizeof(SherpaOnnxOfflineWhisperModelConfig) +
                      sizeof(SherpaOnnxOfflineTdnnModelConfig) + 8 * 4,
              "");
static_assert(sizeof(SherpaOnnxFeatureConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineRecognizerConfig) ==
                  sizeof(SherpaOnnxFeatureConfig) +
                      sizeof(SherpaOnnxOfflineLMConfig) +
                      sizeof(SherpaOnnxOfflineModelConfig) + 4 * 4,
              "");

void PrintOfflineTtsConfig(SherpaOnnxOfflineTtsConfig *tts_config) {
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
  fprintf(stdout, "max num sentences: %d\n", tts_config->max_num_sentences);
}

void PrintOfflineRecognizerConfig(SherpaOnnxOfflineRecognizerConfig *config) {
  auto model_config = &config->model_config;
  auto feat = &config->feat_config;
  auto transducer = &model_config->transducer;
  auto paraformer = &model_config->paraformer;
  auto nemo_ctc = &model_config->nemo_ctc;
  auto whisper = &model_config->whisper;
  auto tdnn = &model_config->tdnn;

  fprintf(stdout, "----------offline transducer model config----------\n");
  fprintf(stdout, "encoder: %s\n", transducer->encoder);
  fprintf(stdout, "decoder: %s\n", transducer->decoder);
  fprintf(stdout, "joiner: %s\n", transducer->joiner);

  fprintf(stdout, "----------offline paraformer model config----------\n");
  fprintf(stdout, "model: %s\n", paraformer->model);

  fprintf(stdout, "----------offline nemo_ctc model config----------\n");
  fprintf(stdout, "model: %s\n", nemo_ctc->model);

  fprintf(stdout, "----------offline whisper model config----------\n");
  fprintf(stdout, "encoder: %s\n", whisper->encoder);
  fprintf(stdout, "decoder: %s\n", whisper->decoder);
  fprintf(stdout, "language: %s\n", whisper->language);
  fprintf(stdout, "task: %s\n", whisper->task);
  fprintf(stdout, "tail_paddings: %d\n", whisper->tail_paddings);

  fprintf(stdout, "----------offline tdnn model config----------\n");
  fprintf(stdout, "model: %s\n", tdnn->model);

  fprintf(stdout, "tokens: %s\n", model_config->tokens);
  fprintf(stdout, "num_threads: %d\n", model_config->num_threads);
  fprintf(stdout, "provider: %s\n", model_config->provider);
  fprintf(stdout, "debug: %d\n", model_config->debug);
  fprintf(stdout, "model type: %s\n", model_config->model_type);
  fprintf(stdout, "modeling unit: %s\n", model_config->modeling_unit);
  fprintf(stdout, "bpe vocab: %s\n", model_config->bpe_vocab);
  fprintf(stdout, "telespeech_ctc: %s\n", model_config->telespeech_ctc);

  fprintf(stdout, "----------feat config----------\n");
  fprintf(stdout, "sample rate: %d\n", feat->sample_rate);
  fprintf(stdout, "feat dim: %d\n", feat->feature_dim);

  fprintf(stdout, "----------recognizer config----------\n");
  fprintf(stdout, "decoding method: %s\n", config->decoding_method);
  fprintf(stdout, "max active paths: %d\n", config->max_active_paths);
  fprintf(stdout, "hotwords_file: %s\n", config->hotwords_file);
  fprintf(stdout, "hotwords_score: %.2f\n", config->hotwords_score);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
