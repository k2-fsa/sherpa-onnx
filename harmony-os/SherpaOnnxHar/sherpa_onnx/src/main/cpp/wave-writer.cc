// scripts/node-addon-api/src/wave-writer.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <sstream>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

// (filename, {samples: samples, sampleRate: sampleRate}
static Napi::Boolean WriteWaveWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsString()) {
    Napi::TypeError::New(env, "Argument 0 should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

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

  int32_t ok =
      SherpaOnnxWriteWave(samples.Data(), samples.ElementLength(), sample_rate,
                          info[0].As<Napi::String>().Utf8Value().c_str());

  return Napi::Boolean::New(env, ok);
}

void InitWaveWriter(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "writeWave"),
              Napi::Function::New(env, WriteWaveWrapper));
}
