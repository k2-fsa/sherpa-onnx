// scripts/node-addon-api/src/streaming-speech-denoiser.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include <algorithm>
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig
GetOnlineSpeechDenoiserGtcrnModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("gtcrn") || !obj.Get("gtcrn").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("gtcrn").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  return c;
}

static SherpaOnnxOfflineSpeechDenoiserDpdfNetModelConfig
GetOnlineSpeechDenoiserDpdfNetModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineSpeechDenoiserDpdfNetModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("dpdfnet") || !obj.Get("dpdfnet").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("dpdfnet").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  return c;
}

static SherpaOnnxOnlineSpeechDenoiserConfig GetOnlineSpeechDenoiserConfig(
    Napi::Object obj) {
  SherpaOnnxOnlineSpeechDenoiserConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("model") || !obj.Get("model").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("model").As<Napi::Object>();

  c.model.gtcrn = GetOnlineSpeechDenoiserGtcrnModelConfig(o);
  c.model.dpdfnet = GetOnlineSpeechDenoiserDpdfNetModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_INT32(num_threads, numThreads);

  if (o.Has("debug") &&
      (o.Get("debug").IsNumber() || o.Get("debug").IsBoolean())) {
    if (o.Get("debug").IsBoolean()) {
      c.model.debug = o.Get("debug").As<Napi::Boolean>().Value();
    } else {
      c.model.debug = o.Get("debug").As<Napi::Number>().Int32Value();
    }
  }

  SHERPA_ONNX_ASSIGN_ATTR_STR(provider, provider);

  return c;
}

static Napi::Object CreateDenoisedAudioObject(
    Napi::Env env, const SherpaOnnxDenoisedAudio *audio,
    bool enable_external_buffer) {
  Napi::Object ans = Napi::Object::New(env);

  if (!audio) {
    ans.Set(Napi::String::New(env, "samples"),
            Napi::Float32Array::New(env, 0));
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

static Napi::External<SherpaOnnxOnlineSpeechDenoiser>
CreateOnlineSpeechDenoiserWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "Expect a single config object")
        .ThrowAsJavaScriptException();
    return {};
  }

  SherpaOnnxOnlineSpeechDenoiserConfig c;
  memset(&c, 0, sizeof(c));
  c = GetOnlineSpeechDenoiserConfig(info[0].As<Napi::Object>());

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      SherpaOnnxCreateOnlineSpeechDenoiser(&c);

  SHERPA_ONNX_DELETE_C_STR(c.model.gtcrn.model);
  SHERPA_ONNX_DELETE_C_STR(c.model.provider);
  SHERPA_ONNX_DELETE_C_STR(c.model.dpdfnet.model);

  if (!sd) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();
    return {};
  }

  return Napi::External<SherpaOnnxOnlineSpeechDenoiser>::New(
      env, const_cast<SherpaOnnxOnlineSpeechDenoiser *>(sd),
      [](Napi::Env /*env*/, SherpaOnnxOnlineSpeechDenoiser *sd) {
        SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
      });
}

static Napi::Object OnlineSpeechDenoiserRunWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsExternal() || !info[1].IsObject()) {
    Napi::TypeError::New(env, "Expect a denoiser handle and an audio object")
        .ThrowAsJavaScriptException();
    return {};
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("samples") || !obj.Get("samples").IsTypedArray()) {
    Napi::TypeError::New(env, "The argument object should have a typed array field samples")
        .ThrowAsJavaScriptException();
    return {};
  }

  if (!obj.Has("sampleRate") || !obj.Get("sampleRate").IsNumber()) {
    Napi::TypeError::New(env, "The argument object should have a number field sampleRate")
        .ThrowAsJavaScriptException();
    return {};
  }

  Napi::Float32Array samples = obj.Get("samples").As<Napi::Float32Array>();
  int32_t sample_rate = obj.Get("sampleRate").As<Napi::Number>().Int32Value();
  bool enable_external_buffer = true;
  if (obj.Has("enableExternalBuffer") &&
      obj.Get("enableExternalBuffer").IsBoolean()) {
    enable_external_buffer =
        obj.Get("enableExternalBuffer").As<Napi::Boolean>().Value();
  }

  const SherpaOnnxDenoisedAudio *audio = SherpaOnnxOnlineSpeechDenoiserRun(
      sd, samples.Data(), samples.ElementLength(), sample_rate);
  return CreateDenoisedAudioObject(env, audio, enable_external_buffer);
}

static Napi::Object OnlineSpeechDenoiserFlushWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expect an online speech denoiser pointer.")
        .ThrowAsJavaScriptException();
    return {};
  }

  bool enable_external_buffer = true;
  if (info.Length() > 1 && info[1].IsBoolean()) {
    enable_external_buffer = info[1].As<Napi::Boolean>().Value();
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  const SherpaOnnxDenoisedAudio *audio =
      SherpaOnnxOnlineSpeechDenoiserFlush(sd);
  return CreateDenoisedAudioObject(env, audio, enable_external_buffer);
}

static void OnlineSpeechDenoiserResetWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expect an online speech denoiser pointer.")
        .ThrowAsJavaScriptException();
    return;
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  SherpaOnnxOnlineSpeechDenoiserReset(sd);
}

static Napi::Number OnlineSpeechDenoiserGetSampleRateWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expect an online speech denoiser pointer.")
        .ThrowAsJavaScriptException();
    return {};
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  return Napi::Number::New(
      env, SherpaOnnxOnlineSpeechDenoiserGetSampleRate(sd));
}

static Napi::Number OnlineSpeechDenoiserGetFrameShiftInSamplesWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expect an online speech denoiser pointer.")
        .ThrowAsJavaScriptException();
    return {};
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  return Napi::Number::New(
      env, SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(sd));
}

void InitOnlineSpeechDenoiser(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOnlineSpeechDenoiser"),
              Napi::Function::New(env, CreateOnlineSpeechDenoiserWrapper));
  exports.Set(Napi::String::New(env, "onlineSpeechDenoiserRunWrapper"),
              Napi::Function::New(env, OnlineSpeechDenoiserRunWrapper));
  exports.Set(Napi::String::New(env, "onlineSpeechDenoiserFlushWrapper"),
              Napi::Function::New(env, OnlineSpeechDenoiserFlushWrapper));
  exports.Set(Napi::String::New(env, "onlineSpeechDenoiserResetWrapper"),
              Napi::Function::New(env, OnlineSpeechDenoiserResetWrapper));
  exports.Set(
      Napi::String::New(env, "onlineSpeechDenoiserGetSampleRateWrapper"),
      Napi::Function::New(env, OnlineSpeechDenoiserGetSampleRateWrapper));
  exports.Set(
      Napi::String::New(env, "onlineSpeechDenoiserGetFrameShiftInSamplesWrapper"),
      Napi::Function::New(env,
                          OnlineSpeechDenoiserGetFrameShiftInSamplesWrapper));
}
