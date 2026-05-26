// c-api-examples/online-speech-enhancement-dpdfnet-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation
//
// We assume you have pre-downloaded model
// from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
// or
// https://huggingface.co/Ceva-IP/DPDFNet
//
// An example command to download
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
*/
// clang-format on

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static int32_t AppendSamples(float **samples, int32_t *num_samples,
                             const SherpaOnnxDenoisedAudio *audio) {
  float *p = NULL;

  if (!audio || audio->n == 0) {
    return 1;
  }

  p = (float *)realloc(*samples, sizeof(float) * (*num_samples + audio->n));
  if (!p) {
    fprintf(stderr, "Failed to allocate memory for output samples\n");
    return 0;
  }

  memcpy(p + *num_samples, audio->samples, sizeof(float) * audio->n);
  *samples = p;
  *num_samples += audio->n;
  return 1;
}

int32_t main() {
  SherpaOnnxOnlineSpeechDenoiserConfig config;
  const char *model_filename = "./dpdfnet_baseline.onnx";
  const char *wav_filename = "./inp_16k.wav";
  const char *out_wave_filename = "./enhanced-online-dpdfnet.wav";
  float *samples = NULL;
  int32_t num_samples = 0;

  memset(&config, 0, sizeof(config));
  config.model.dpdfnet.model = model_filename;

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      SherpaOnnxCreateOnlineSpeechDenoiser(&config);
  if (!sd) {
    fprintf(stderr, "Please check your config\n");
    return -1;
  }

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (!wave) {
    SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  int32_t frame_shift = SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(sd);
  for (int32_t start = 0; start < wave->num_samples; start += frame_shift) {
    int32_t n = frame_shift;
    if (start + n > wave->num_samples) {
      n = wave->num_samples - start;
    }

    const SherpaOnnxDenoisedAudio *audio = SherpaOnnxOnlineSpeechDenoiserRun(
        sd, wave->samples + start, n, wave->sample_rate);
    int32_t ok = AppendSamples(&samples, &num_samples, audio);
    SherpaOnnxDestroyDenoisedAudio(audio);
    if (!ok) {
      free(samples);
      SherpaOnnxFreeWave(wave);
      SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
      return -1;
    }
  }

  const SherpaOnnxDenoisedAudio *tail = SherpaOnnxOnlineSpeechDenoiserFlush(sd);
  int32_t sample_rate = tail ? tail->sample_rate
                             : SherpaOnnxOnlineSpeechDenoiserGetSampleRate(sd);
  int32_t ok = AppendSamples(&samples, &num_samples, tail);
  SherpaOnnxDestroyDenoisedAudio(tail);
  if (!ok) {
    free(samples);
    SherpaOnnxFreeWave(wave);
    SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
    return -1;
  }

  if (num_samples == 0) {
    fprintf(stderr, "No denoised samples were produced\n");
    free(samples);
    SherpaOnnxFreeWave(wave);
    SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
    return -1;
  }

  SherpaOnnxWriteWave(samples, num_samples, sample_rate, out_wave_filename);

  free(samples);
  SherpaOnnxFreeWave(wave);
  SherpaOnnxDestroyOnlineSpeechDenoiser(sd);

  fprintf(stdout, "Saved to %s\n", out_wave_filename);
  return 0;
}
