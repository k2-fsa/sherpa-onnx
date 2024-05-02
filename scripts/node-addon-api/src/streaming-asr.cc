// scripts/node-addon-api/src/streaming-asr.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <sstream>

#include "napi.h"
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxFeatureConfig GetFeatureConfig(Napi::Object obj) {
  SherpaOnnxFeatureConfig config;
  config.sample_rate = 16000;
  config.feature_dim = 80;

  if (obj.Has("featConfig") && obj.Get("featConfig").IsObject()) {
    Napi::Object featConfig = obj.Get("featConfig").As<Napi::Object>();

    if (featConfig.Has("sampleRate") &&
        featConfig.Get("sampleRate").IsNumber()) {
      config.sample_rate =
          featConfig.Get("sampleRate").As<Napi::Number>().Int32Value();
    }

    if (featConfig.Has("featureDim") &&
        featConfig.Get("featureDim").IsNumber()) {
      config.feature_dim =
          featConfig.Get("featureDim").As<Napi::Number>().Int32Value();
    }
  }

  fprintf(stderr, "sample_rate: %d\n", config.sample_rate);
  fprintf(stderr, "feat_dim: %d\n", config.feature_dim);

  return config;
}

static Napi::External<SherpaOnnxOnlineRecognizer> CreateOnlineRecognizerWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str().c_str()).ThrowAsJavaScriptException();

    return Napi::External<SherpaOnnxOnlineRecognizer>::New(env, nullptr);
  }

  if (!info[0].IsObject()) {
    Napi::TypeError::New(env, "Expect an object as the argument")
        .ThrowAsJavaScriptException();

    return Napi::External<SherpaOnnxOnlineRecognizer>::New(env, nullptr);
  }

  Napi::Object config = info[0].As<Napi::Object>();
  SherpaOnnxOnlineRecognizerConfig c;
  c.feat_config = GetFeatureConfig(config);

  return Napi::External<SherpaOnnxOnlineRecognizer>::New(env, nullptr);
}

void InitStreamingAsr(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOnlineRecognizer"),
              Napi::Function::New(env, CreateOnlineRecognizerWrapper));
}
