// c-api-examples/speech-enhancement-voxlock-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of
// one speaker, e.g., the caller in a phone call, and removes the other
// voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock takes the loudest
// 3 seconds in the first 15 seconds of the input as the speaker to keep. See
// ./speech-enhancement-voxlock-with-voice-sample-c-api.c if you have a voice
// sample of the speaker.
//
// We assume you have pre-downloaded the model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// An example command to download
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
tar xvf sherpa-onnx-voxlock-int8.tar.bz2
*/
// clang-format on
#include <stdio.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  SherpaOnnxOfflineSpeechDenoiserConfig config;
  const char *enroll_model = "./sherpa-onnx-voxlock-int8/enroll.onnx";
  const char *chunk_model = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx";
  const char *wav_filename =
      "./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav";
  const char *out_wave_filename = "./voxlock-enhanced-16k.wav";

  memset(&config, 0, sizeof(config));
  config.model.voxlock.enroll_model = enroll_model;
  config.model.voxlock.chunk_model = chunk_model;

  const SherpaOnnxOfflineSpeechDenoiser *sd =
      SherpaOnnxCreateOfflineSpeechDenoiser(&config);
  if (!sd) {
    fprintf(stderr, "Please check your config");
    return -1;
  }

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    SherpaOnnxDestroyOfflineSpeechDenoiser(sd);
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  const SherpaOnnxDenoisedAudio *denoised = SherpaOnnxOfflineSpeechDenoiserRun(
      sd, wave->samples, wave->num_samples, wave->sample_rate);

  SherpaOnnxWriteWave(denoised->samples, denoised->n, denoised->sample_rate,
                      out_wave_filename);

  SherpaOnnxDestroyDenoisedAudio(denoised);
  SherpaOnnxFreeWave(wave);
  SherpaOnnxDestroyOfflineSpeechDenoiser(sd);

  fprintf(stdout, "Saved to %s\n", out_wave_filename);
}
