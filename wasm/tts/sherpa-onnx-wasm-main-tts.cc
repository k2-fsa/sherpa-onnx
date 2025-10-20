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
static_assert(sizeof(SherpaOnnxOfflineTtsMatchaModelConfig) == 8 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTtsKokoroModelConfig) == 8 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTtsKittenModelConfig) == 5 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTtsZipvoiceModelConfig) == 10 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTtsModelConfig) ==
                  sizeof(SherpaOnnxOfflineTtsVitsModelConfig) +
                      sizeof(SherpaOnnxOfflineTtsMatchaModelConfig) +
                      sizeof(SherpaOnnxOfflineTtsKokoroModelConfig) + 3 * 4 +
                      sizeof(SherpaOnnxOfflineTtsKittenModelConfig) +
                      sizeof(SherpaOnnxOfflineTtsZipvoiceModelConfig),
              "");
static_assert(sizeof(SherpaOnnxOfflineTtsConfig) ==
                  sizeof(SherpaOnnxOfflineTtsModelConfig) + 4 * 4,
              "");

void MyPrint(SherpaOnnxOfflineTtsConfig *tts_config) {
  auto tts_model_config = &tts_config->model;
  auto vits_model_config = &tts_model_config->vits;
  auto matcha_model_config = &tts_model_config->matcha;
  auto kokoro = &tts_model_config->kokoro;
  auto kitten = &tts_model_config->kitten;
  auto zipvoice = &tts_model_config->zipvoice;
  fprintf(stdout, "----------vits model config----------\n");
  fprintf(stdout, "model: %s\n", vits_model_config->model);
  fprintf(stdout, "lexicon: %s\n", vits_model_config->lexicon);
  fprintf(stdout, "tokens: %s\n", vits_model_config->tokens);
  fprintf(stdout, "data_dir: %s\n", vits_model_config->data_dir);
  fprintf(stdout, "noise scale: %.3f\n", vits_model_config->noise_scale);
  fprintf(stdout, "noise scale w: %.3f\n", vits_model_config->noise_scale_w);
  fprintf(stdout, "length scale: %.3f\n", vits_model_config->length_scale);
  fprintf(stdout, "dict_dir: %s\n", vits_model_config->dict_dir);

  fprintf(stdout, "----------matcha model config----------\n");
  fprintf(stdout, "acoustic_model: %s\n", matcha_model_config->acoustic_model);
  fprintf(stdout, "vocoder: %s\n", matcha_model_config->vocoder);
  fprintf(stdout, "lexicon: %s\n", matcha_model_config->lexicon);
  fprintf(stdout, "tokens: %s\n", matcha_model_config->tokens);
  fprintf(stdout, "data_dir: %s\n", matcha_model_config->data_dir);
  fprintf(stdout, "noise scale: %.3f\n", matcha_model_config->noise_scale);
  fprintf(stdout, "length scale: %.3f\n", matcha_model_config->length_scale);
  fprintf(stdout, "dict_dir: %s\n", matcha_model_config->dict_dir);

  fprintf(stdout, "----------kokoro model config----------\n");
  fprintf(stdout, "model: %s\n", kokoro->model);
  fprintf(stdout, "voices: %s\n", kokoro->voices);
  fprintf(stdout, "tokens: %s\n", kokoro->tokens);
  fprintf(stdout, "data_dir: %s\n", kokoro->data_dir);
  fprintf(stdout, "length scale: %.3f\n", kokoro->length_scale);
  fprintf(stdout, "dict_dir: %s\n", kokoro->dict_dir);
  fprintf(stdout, "lexicon: %s\n", kokoro->lexicon);
  fprintf(stdout, "lang: %s\n", kokoro->lang);

  fprintf(stdout, "----------kitten model config----------\n");
  fprintf(stdout, "model: %s\n", kitten->model);
  fprintf(stdout, "voices: %s\n", kitten->voices);
  fprintf(stdout, "tokens: %s\n", kitten->tokens);
  fprintf(stdout, "data_dir: %s\n", kitten->data_dir);
  fprintf(stdout, "length scale: %.3f\n", kitten->length_scale);

  fprintf(stdout, "----------zipvoice model config----------\n");
  fprintf(stdout, "tokens: %s\n", zipvoice->tokens);
  fprintf(stdout, "text_model: %s\n", zipvoice->text_model);
  fprintf(stdout, "flow_matching_model: %s\n", zipvoice->flow_matching_model);
  fprintf(stdout, "vocoder: %s\n", zipvoice->vocoder);
  fprintf(stdout, "data_dir: %s\n", zipvoice->data_dir);
  fprintf(stdout, "pinyin_dict: %s\n", zipvoice->pinyin_dict);
  fprintf(stdout, "feat scale: %.3f\n", zipvoice->feat_scale);
  fprintf(stdout, "t_shift: %.3f\n", zipvoice->t_shift);
  fprintf(stdout, "target_rms: %.3f\n", zipvoice->target_rms);
  fprintf(stdout, "guidance_scale: %.3f\n", zipvoice->guidance_scale);

  fprintf(stdout, "----------tts model config----------\n");
  fprintf(stdout, "num threads: %d\n", tts_model_config->num_threads);
  fprintf(stdout, "debug: %d\n", tts_model_config->debug);
  fprintf(stdout, "provider: %s\n", tts_model_config->provider);

  fprintf(stdout, "----------tts config----------\n");
  fprintf(stdout, "rule_fsts: %s\n", tts_config->rule_fsts);
  fprintf(stdout, "rule_fars: %s\n", tts_config->rule_fars);
  fprintf(stdout, "max num sentences: %d\n", tts_config->max_num_sentences);
  fprintf(stdout, "silence scale: %.3f\n", tts_config->silence_scale);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
