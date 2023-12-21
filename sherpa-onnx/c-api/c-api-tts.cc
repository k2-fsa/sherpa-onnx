// sherpa-onnx/c-api/c-api-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/c-api/c-api.h"

#include <cstdio>
#include <memory>

#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/csrc/wave-writer.h"

#define SHERPA_ONNX_OR(x, y) (x ? x : y)

struct SherpaOnnxOfflineTts {
  std::unique_ptr<sherpa_onnx::OfflineTts> impl;
};

SherpaOnnxOfflineTts *SherpaOnnxCreateOfflineTts(
    const SherpaOnnxOfflineTtsConfig *config) {
  sherpa_onnx::OfflineTtsConfig tts_config;

  tts_config.model.vits.model = SHERPA_ONNX_OR(config->model.vits.model, "");
  tts_config.model.vits.lexicon =
      SHERPA_ONNX_OR(config->model.vits.lexicon, "");
  tts_config.model.vits.tokens = SHERPA_ONNX_OR(config->model.vits.tokens, "");
  tts_config.model.vits.data_dir =
      SHERPA_ONNX_OR(config->model.vits.data_dir, "");
  tts_config.model.vits.noise_scale =
      SHERPA_ONNX_OR(config->model.vits.noise_scale, 0.667);
  tts_config.model.vits.noise_scale_w =
      SHERPA_ONNX_OR(config->model.vits.noise_scale_w, 0.8);
  tts_config.model.vits.length_scale =
      SHERPA_ONNX_OR(config->model.vits.length_scale, 1.0);

  tts_config.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  tts_config.model.debug = config->model.debug;
  tts_config.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");
  tts_config.rule_fsts = SHERPA_ONNX_OR(config->rule_fsts, "");
  tts_config.max_num_sentences = SHERPA_ONNX_OR(config->max_num_sentences, 2);

  if (tts_config.model.debug) {
    fprintf(stderr, "%s\n", tts_config.ToString().c_str());
  }

  SherpaOnnxOfflineTts *tts = new SherpaOnnxOfflineTts;

  tts->impl = std::make_unique<sherpa_onnx::OfflineTts>(tts_config);

  return tts;
}

void SherpaOnnxDestroyOfflineTts(SherpaOnnxOfflineTts *tts) { delete tts; }

int32_t SherpaOnnxOfflineTtsSampleRate(const SherpaOnnxOfflineTts *tts) {
  return tts->impl->SampleRate();
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerate(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid,
    float speed) {
  return SherpaOnnxOfflineTtsGenerateWithCallback(tts, text, sid, speed,
                                                  nullptr);
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateWithCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallback callback) {
  sherpa_onnx::GeneratedAudio audio =
      tts->impl->Generate(text, sid, speed, callback);

  if (audio.samples.empty()) {
    return nullptr;
  }

  SherpaOnnxGeneratedAudio *ans = new SherpaOnnxGeneratedAudio;

  float *samples = new float[audio.samples.size()];
  std::copy(audio.samples.begin(), audio.samples.end(), samples);

  ans->samples = samples;
  ans->n = audio.samples.size();
  ans->sample_rate = audio.sample_rate;

  return ans;
}

void SherpaOnnxDestroyOfflineTtsGeneratedAudio(
    const SherpaOnnxGeneratedAudio *p) {
  if (p) {
    delete[] p->samples;
    delete p;
  }
}

int32_t SherpaOnnxWriteWave(const float *samples, int32_t n,
                            int32_t sample_rate, const char *filename) {
  return sherpa_onnx::WriteWave(filename, sample_rate, samples, n);
}
