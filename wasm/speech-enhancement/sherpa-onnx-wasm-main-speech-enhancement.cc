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
static_assert(sizeof(SherpaOnnxOfflineSpeechDenoiserDpdfNetModelConfig) ==
                  2 * 4,
              "");
static_assert(sizeof(SherpaOnnxOfflineSpeechDenoiserVoxLockModelConfig) ==
                  3 * 4,
              "");
static_assert(
    sizeof(SherpaOnnxOfflineSpeechDenoiserModelConfig) ==
        sizeof(SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig) +
            sizeof(SherpaOnnxOfflineSpeechDenoiserDpdfNetModelConfig) +
            sizeof(SherpaOnnxOfflineSpeechDenoiserVoxLockModelConfig) + 3 * 4,
    "");
static_assert(sizeof(SherpaOnnxOfflineSpeechDenoiserConfig) ==
                  sizeof(SherpaOnnxOfflineSpeechDenoiserModelConfig),
              "");

void MyPrint(SherpaOnnxOfflineSpeechDenoiserConfig *config) {
  auto model = &config->model;
  auto gtcrn = &model->gtcrn;
  auto dpdfnet = &model->dpdfnet;
  auto voxlock = &model->voxlock;
  fprintf(stdout, "----------offline speech denoiser model config----------\n");
  fprintf(stdout, "gtcrn: %s\n", gtcrn->model);
  fprintf(stdout, "dpdfnet: %s\n", dpdfnet->model);
  fprintf(stdout, "voxlock enroll model: %s\n", voxlock->enroll_model);
  fprintf(stdout, "voxlock chunk model: %s\n", voxlock->chunk_model);
  fprintf(stdout, "voxlock voice sample: %s\n", voxlock->voice_sample);
  fprintf(stdout, "num threads: %d\n", model->num_threads);
  fprintf(stdout, "debug: %d\n", model->debug);
  fprintf(stdout, "provider: %s\n", model->provider);
  fprintf(stdout, "dpdfnet attenuation limit (dB): %.2f\n",
          model->dpdfnet.attenuation_limit_db);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
