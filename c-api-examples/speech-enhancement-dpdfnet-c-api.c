// c-api-examples/speech-enhancement-dpdfnet-c-api.c
//
// Copyright (c)  2025  Xiaomi Corporation
//
// We assume you have pre-downloaded model
// from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
// or
// https://huggingface.co/Ceva-IP/DPDFNet
//
//
// An example command to download
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet4.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet8.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2_48khz_hr.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
*/
// clang-format on
//
// Use dpdfnet_baseline.onnx, dpdfnet2.onnx, dpdfnet4.onnx, or dpdfnet8.onnx
// for 16 kHz downstream ASR or speech recognition.
// Use dpdfnet2_48khz_hr.onnx for 48 kHz enhancement output.
#include <stdio.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  SherpaOnnxOfflineSpeechDenoiserConfig config;
  const char *model_filename = "./dpdfnet_baseline.onnx";
  const char *wav_filename = "./inp_16k.wav";
  const char *out_wave_filename = "./enhanced.wav";

  memset(&config, 0, sizeof(config));
  config.model.dpdfnet.model = model_filename;

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
