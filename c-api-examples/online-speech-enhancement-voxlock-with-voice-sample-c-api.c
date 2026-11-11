// c-api-examples/online-speech-enhancement-voxlock-with-voice-sample-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of
// one speaker, e.g., the caller in a phone call, and removes the other
// voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example provides a voice sample of the speaker to keep, so VoxLock
// isolates that speaker from the first sample. The voice sample must have at
// least 3 seconds of the speaker's voice; from a longer file, VoxLock uses
// the loudest 3 seconds within its first 30 seconds. See
// ./online-speech-enhancement-voxlock-c-api.c if you don't have a voice
// sample.
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
  const char *enroll_model = "./sherpa-onnx-voxlock-int8/enroll.onnx";
  const char *chunk_model = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx";
  const char *voice_sample =
      "./sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav";
  const char *wav_filename =
      "./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav";
  const char *out_wave_filename =
      "./voxlock-enhanced-online-with-voice-sample-16k.wav";
  float *samples = NULL;
  int32_t num_samples = 0;

  memset(&config, 0, sizeof(config));
  config.model.voxlock.enroll_model = enroll_model;
  config.model.voxlock.chunk_model = chunk_model;
  config.model.voxlock.voice_sample = voice_sample;

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
