// c-api-examples/source-separation-spleeter-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation

// This file demonstrates how to use the source-separation C API
// with the Spleeter 2-stems model.
//
// Usage:
//
// 1. Download the test model and audio
//
//  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2
//  tar xjf sherpa-onnx-spleeter-2stems-fp16.tar.bz2
//
//  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
//
// 2. Build
//
//  cmake -DSHERPA_ONNX_ENABLE_C_API=ON ..
//  make source-separation-spleeter-c-api
//
// 3. Run
//
//  ./bin/source-separation-spleeter-c-api

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  SherpaOnnxOfflineSourceSeparationConfig config;
  memset(&config, 0, sizeof(config));
  config.model.spleeter.vocals =
      "./sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx";
  config.model.spleeter.accompaniment =
      "./sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx";
  config.model.num_threads = 1;

  const SherpaOnnxOfflineSourceSeparation *ss =
      SherpaOnnxCreateOfflineSourceSeparation(&config);
  if (!ss) {
    fprintf(stderr, "Failed to create source separation engine\n");
    return -1;
  }

  const SherpaOnnxMultiChannelWave *wave =
      SherpaOnnxReadWaveMultiChannel("./qi-feng-le-zh.wav");
  if (!wave) {
    fprintf(stderr, "Failed to read ./qi-feng-le-zh.wav\n");
    SherpaOnnxDestroyOfflineSourceSeparation(ss);
    return -1;
  }

  fprintf(stdout, "Input wave: channels=%d, samples_per_channel=%d, sample_rate=%d\n",
          wave->num_channels, wave->num_samples, wave->sample_rate);

  int32_t num_channels = wave->num_channels;

  const SherpaOnnxSourceSeparationOutput *output =
      SherpaOnnxOfflineSourceSeparationProcess(
          ss, wave->samples, num_channels, wave->num_samples,
          wave->sample_rate);

  if (!output) {
    fprintf(stderr, "Source separation failed\n");
    SherpaOnnxFreeMultiChannelWave(wave);
    SherpaOnnxDestroyOfflineSourceSeparation(ss);
    return -1;
  }

  fprintf(stdout, "Output: %d stems, sample_rate=%d\n", output->num_stems,
          output->sample_rate);

  // Write each stem to a separate multi-channel wave file.
  const char *stem_names[] = {"vocals", "accompaniment"};
  for (int32_t s = 0;
       s < output->num_stems &&
       s < (int32_t)(sizeof(stem_names) / sizeof(stem_names[0]));
       ++s) {
    int32_t nc = output->stems[s].num_channels;
    int32_t ns = output->stems[s].n;
    char filename[256];
    snprintf(filename, sizeof(filename), "%s.wav", stem_names[s]);
    SherpaOnnxWriteWaveMultiChannel((const float *const *)output->stems[s].samples, ns,
                                    output->sample_rate, nc, filename);
    fprintf(stdout, "Saved %s (%d channels, %d samples, %d Hz)\n", filename,
            nc, ns, output->sample_rate);
  }

  // Cleanup
  SherpaOnnxDestroySourceSeparationOutput(output);
  SherpaOnnxFreeMultiChannelWave(wave);
  SherpaOnnxDestroyOfflineSourceSeparation(ss);

  return 0;
}
