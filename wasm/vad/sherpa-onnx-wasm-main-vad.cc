// wasm/sherpa-onnx-wasm-main-vad.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <stdio.h>

#include <algorithm>
#include <memory>

#include "sherpa-onnx/c-api/c-api.h"

// see also
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html

extern "C" {

static_assert(sizeof(SherpaOnnxSileroVadModelConfig) == 6 * 4, "");

static_assert(sizeof(SherpaOnnxVadModelConfig) ==
                  sizeof(SherpaOnnxSileroVadModelConfig) + 4 * 4,
              "");
void MyPrint(SherpaOnnxVadModelConfig *config) {
  auto silero_vad = &config->silero_vad;

  fprintf(stdout, "----------silero_vad config----------\n");
  fprintf(stdout, "model: %s\n", silero_vad->model);
  fprintf(stdout, "threshold: %.3f\n", silero_vad->threshold);
  fprintf(stdout, "min_silence_duration: %.3f\n",
          silero_vad->min_silence_duration);
  fprintf(stdout, "min_speech_duration: %.3f\n",
          silero_vad->min_speech_duration);
  fprintf(stdout, "window_size: %d\n", silero_vad->window_size);
  fprintf(stdout, "max_speech_duration: %.3f\n",
          silero_vad->max_speech_duration);

  fprintf(stdout, "----------config----------\n");

  fprintf(stdout, "sample_rate: %d\n", config->sample_rate);
  fprintf(stdout, "num_threads: %d\n", config->num_threads);

  fprintf(stdout, "provider: %s\n", config->provider);
  fprintf(stdout, "debug: %d\n", config->debug);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
