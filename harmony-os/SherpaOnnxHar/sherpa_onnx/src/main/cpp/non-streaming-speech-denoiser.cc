// scripts/node-addon-api/src/non-streaming-speech-denoiser.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include <algorithm>
#include <memory>
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig
GetOfflineSpeechDenoiserGtcrnModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("gtcrn") || !obj.Get("gtcrn").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("gtcrn").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);

  return c;
}

static SherpaOnnxOfflineSpeechDenoiserModelConfig
GetOfflineSpeechDenoiserModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineSpeechDenoiserModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("model") || !obj.Get("model").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("model").As<Napi::Object>();

  c.gtcrn = GetOfflineSpeechDenoiserGtcrnModelConfig(o);

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

static Napi::External<SherpaOnnxOfflineSpeechDenoiser>
CreateOfflineSpeechDenoiserWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
#if __OHOS__
  // the last argument is the NativeResourceManager
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

  Napi::Object o = info[0].As<Napi::Object>();

  SherpaOnnxOfflineSpeechDenoiserConfig c;
  memset(&c, 0, sizeof(c));
  c.model = GetOfflineSpeechDenoiserModelConfig(o);

#if __OHOS__
  std::unique_ptr<NativeResourceManager,
                  decltype(&OH_ResourceManager_ReleaseNativeResourceManager)>
      mgr(OH_ResourceManager_InitNativeResourceManager(env, info[1]),
          &OH_ResourceManager_ReleaseNativeResourceManager);

  const SherpaOnnxOfflineSpeechDenoiser *sd =
      SherpaOnnxCreateOfflineSpeechDenoiserOHOS(&c, mgr.get());
#else
  const SherpaOnnxOfflineSpeechDenoiser *sd =
      SherpaOnnxCreateOfflineSpeechDenoiser(&c);
#endif

  SHERPA_ONNX_DELETE_C_STR(c.model.gtcrn.model);
  SHERPA_ONNX_DELETE_C_STR(c.model.provider);

  if (!sd) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxOfflineSpeechDenoiser>::New(
      env, const_cast<SherpaOnnxOfflineSpeechDenoiser *>(sd),
      [](Napi::Env env, SherpaOnnxOfflineSpeechDenoiser *sd) {
        SherpaOnnxDestroyOfflineSpeechDenoiser(sd);
      });
}

static Napi::Object OfflineSpeechDenoiserRunWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(
        env, "Argument 0 should be an offline speech denoiser pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOfflineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOfflineSpeechDenoiser>>().Data();

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Argument 1 should be an object")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("samples")) {
    Napi::TypeError::New(env, "The argument object should have a field samples")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("samples").IsTypedArray()) {
    Napi::TypeError::New(env, "The object['samples'] should be a typed array")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("sampleRate")) {
    Napi::TypeError::New(env,
                         "The argument object should have a field sampleRate")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("sampleRate").IsNumber()) {
    Napi::TypeError::New(env, "The object['samples'] should be a number")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Float32Array samples = obj.Get("samples").As<Napi::Float32Array>();
  int32_t sample_rate = obj.Get("sampleRate").As<Napi::Number>().Int32Value();

  const SherpaOnnxDenoisedAudio *audio;

#if __OHOS__
  // Note(fangjun): For unknown reasons on HarmonyOS, we need to divide it by
  // sizeof(float) here
  audio = SherpaOnnxOfflineSpeechDenoiserRun(
      sd, samples.Data(), samples.ElementLength() / sizeof(float), sample_rate);
#else
  audio = SherpaOnnxOfflineSpeechDenoiserRun(
      sd, samples.Data(), samples.ElementLength(), sample_rate);
#endif

  bool enable_external_buffer = true;
  if (obj.Has("enableExternalBuffer") &&
      obj.Get("enableExternalBuffer").IsBoolean()) {
    enable_external_buffer =
        obj.Get("enableExternalBuffer").As<Napi::Boolean>().Value();
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

    Napi::Object ans = Napi::Object::New(env);
    ans.Set(Napi::String::New(env, "samples"), float32Array);
    ans.Set(Napi::String::New(env, "sampleRate"), audio->sample_rate);
    return ans;
  } else {
    // don't use external buffer
    Napi::ArrayBuffer arrayBuffer =
        Napi::ArrayBuffer::New(env, sizeof(float) * audio->n);

    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, audio->n, arrayBuffer, 0);

    std::copy(audio->samples, audio->samples + audio->n, float32Array.Data());

    Napi::Object ans = Napi::Object::New(env);
    ans.Set(Napi::String::New(env, "samples"), float32Array);
    ans.Set(Napi::String::New(env, "sampleRate"), audio->sample_rate);
    SherpaOnnxDestroyDenoisedAudio(audio);
    return ans;
  }
}

static Napi::Number OfflineSpeechDenoiserGetSampleRateWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(
        env, "Argument 0 should be an offline speech denoiser pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOfflineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOfflineSpeechDenoiser>>().Data();

  int32_t sample_rate = SherpaOnnxOfflineSpeechDenoiserGetSampleRate(sd);

  return Napi::Number::New(env, sample_rate);
}

void InitNonStreamingSpeechDenoiser(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOfflineSpeechDenoiser"),
              Napi::Function::New(env, CreateOfflineSpeechDenoiserWrapper));

  exports.Set(Napi::String::New(env, "offlineSpeechDenoiserRunWrapper"),
              Napi::Function::New(env, OfflineSpeechDenoiserRunWrapper));

  exports.Set(
      Napi::String::New(env, "offlineSpeechDenoiserGetSampleRateWrapper"),
      Napi::Function::New(env, OfflineSpeechDenoiserGetSampleRateWrapper));
}
