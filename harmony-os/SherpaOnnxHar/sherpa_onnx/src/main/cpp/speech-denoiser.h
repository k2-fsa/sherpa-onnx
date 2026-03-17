// scripts/node-addon-api/src/speech-denoiser.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_HARMONY_OS_SHERPAONNXHAR_SHERPA_ONNX_SRC_MAIN_CPP_SPEECH_DENOISER_H_
#define SHERPA_ONNX_HARMONY_OS_SHERPAONNXHAR_SHERPA_ONNX_SRC_MAIN_CPP_SPEECH_DENOISER_H_

#include <algorithm>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static inline SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig
GetSpeechDenoiserGtcrnModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("gtcrn") || !obj.Get("gtcrn").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("gtcrn").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  return c;
}

static inline SherpaOnnxOfflineSpeechDenoiserDpdfNetModelConfig
GetSpeechDenoiserDpdfNetModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineSpeechDenoiserDpdfNetModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("dpdfnet") || !obj.Get("dpdfnet").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("dpdfnet").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  return c;
}

static inline SherpaOnnxOfflineSpeechDenoiserModelConfig
GetSpeechDenoiserModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineSpeechDenoiserModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("model") || !obj.Get("model").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("model").As<Napi::Object>();
  c.gtcrn = GetSpeechDenoiserGtcrnModelConfig(o);
  c.dpdfnet = GetSpeechDenoiserDpdfNetModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_INT32(num_threads, numThreads);

  if (o.Has("debug") &&
      (o.Get("debug").IsNumber() || o.Get("debug").IsBoolean())) {
    if (o.Get("debug").IsBoolean()) {
      c.debug = o.Get("debug").As<Napi::Boolean>().Value();
    } else {
      c.debug = o.Get("debug").As<Napi::Number>().Int32Value();
    }
  }

  SHERPA_ONNX_ASSIGN_ATTR_STR(provider, provider);

  return c;
}

static inline void DeleteSpeechDenoiserModelConfig(
    const SherpaOnnxOfflineSpeechDenoiserModelConfig &c) {
  SHERPA_ONNX_DELETE_C_STR(c.gtcrn.model);
  SHERPA_ONNX_DELETE_C_STR(c.provider);
  SHERPA_ONNX_DELETE_C_STR(c.dpdfnet.model);
}

static inline bool GetEnableExternalBuffer(Napi::Object obj) {
  if (obj.Has("enableExternalBuffer") &&
      obj.Get("enableExternalBuffer").IsBoolean()) {
    return obj.Get("enableExternalBuffer").As<Napi::Boolean>().Value();
  }

  return true;
}

static inline int32_t GetFloat32ArrayElementLength(Napi::Float32Array samples) {
#if __OHOS__
  return samples.ElementLength() / sizeof(float);
#else
  return samples.ElementLength();
#endif
}

static inline Napi::Object CreateDenoisedAudioObject(
    Napi::Env env, const SherpaOnnxDenoisedAudio *audio,
    bool enable_external_buffer) {
  Napi::Object ans = Napi::Object::New(env);

  if (!audio) {
    ans.Set(Napi::String::New(env, "samples"), Napi::Float32Array::New(env, 0));
    ans.Set(Napi::String::New(env, "sampleRate"), 0);
    return ans;
  }

  if (enable_external_buffer) {
    Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
        env, const_cast<float *>(audio->samples), sizeof(float) * audio->n,
        [](Napi::Env /*env*/, void * /*data*/,
           const SherpaOnnxDenoisedAudio *hint) {
          SherpaOnnxDestroyDenoisedAudio(hint);
        },
        audio);
    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, audio->n, arrayBuffer, 0);
    ans.Set(Napi::String::New(env, "samples"), float32Array);
    ans.Set(Napi::String::New(env, "sampleRate"), audio->sample_rate);
    return ans;
  }

  Napi::ArrayBuffer arrayBuffer =
      Napi::ArrayBuffer::New(env, sizeof(float) * audio->n);
  Napi::Float32Array float32Array =
      Napi::Float32Array::New(env, audio->n, arrayBuffer, 0);

  if (audio->n > 0 && audio->samples) {
    std::copy(audio->samples, audio->samples + audio->n, float32Array.Data());
  }

  ans.Set(Napi::String::New(env, "samples"), float32Array);
  ans.Set(Napi::String::New(env, "sampleRate"), audio->sample_rate);
  SherpaOnnxDestroyDenoisedAudio(audio);
  return ans;
}

#endif  // SHERPA_ONNX_HARMONY_OS_SHERPAONNXHAR_SHERPA_ONNX_SRC_MAIN_CPP_SPEECH_DENOISER_H_
