// scripts/node-addon-api/src/streaming-speech-denoiser.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include <memory>
#include <sstream>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"
#include "speech-denoiser.h"  // NOLINT

static Napi::External<SherpaOnnxOnlineSpeechDenoiser>
CreateOnlineSpeechDenoiserWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
#if __OHOS__
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();
    return {};
  }
#else
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();
    return {};
  }
#endif

  if (!info[0].IsObject()) {
    Napi::TypeError::New(env, "Expect an object as the argument")
        .ThrowAsJavaScriptException();
    return {};
  }

  SherpaOnnxOnlineSpeechDenoiserConfig c;
  memset(&c, 0, sizeof(c));
  c.model = GetSpeechDenoiserModelConfig(info[0].As<Napi::Object>());

#if __OHOS__
  std::unique_ptr<NativeResourceManager,
                  decltype(&OH_ResourceManager_ReleaseNativeResourceManager)>
      mgr(OH_ResourceManager_InitNativeResourceManager(env, info[1]),
          &OH_ResourceManager_ReleaseNativeResourceManager);

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      SherpaOnnxCreateOnlineSpeechDenoiserOHOS(&c, mgr.get());
#else
  const SherpaOnnxOnlineSpeechDenoiser *sd =
      SherpaOnnxCreateOnlineSpeechDenoiser(&c);
#endif

  DeleteSpeechDenoiserModelConfig(c.model);

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
    Napi::TypeError::New(
        env, "The argument object should have a typed array field samples")
        .ThrowAsJavaScriptException();
    return {};
  }

  if (!obj.Has("sampleRate") || !obj.Get("sampleRate").IsNumber()) {
    Napi::TypeError::New(
        env, "The argument object should have a number field sampleRate")
        .ThrowAsJavaScriptException();
    return {};
  }

  Napi::Float32Array samples = obj.Get("samples").As<Napi::Float32Array>();
  int32_t sample_rate = obj.Get("sampleRate").As<Napi::Number>().Int32Value();
  const SherpaOnnxDenoisedAudio *audio = SherpaOnnxOnlineSpeechDenoiserRun(
      sd, samples.Data(), GetFloat32ArrayElementLength(samples), sample_rate);
  return CreateDenoisedAudioObject(env, audio, GetEnableExternalBuffer(obj));
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
  return Napi::Number::New(env,
                           SherpaOnnxOnlineSpeechDenoiserGetSampleRate(sd));
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
  exports.Set(Napi::String::New(
                  env, "onlineSpeechDenoiserGetFrameShiftInSamplesWrapper"),
              Napi::Function::New(
                  env, OnlineSpeechDenoiserGetFrameShiftInSamplesWrapper));
}
