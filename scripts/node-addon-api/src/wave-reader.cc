// scripts/node-addon-api/src/wave-reader.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <sstream>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static Napi::Object ReadWaveWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }
  if (!info[0].IsString()) {
    Napi::TypeError::New(env, "Argument should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

  std::string filename = info[0].As<Napi::String>().Utf8Value();

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(filename.c_str());
  if (!wave) {
    std::ostringstream os;
    os << "Failed to read '" << filename << "'";
    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
      env, const_cast<float *>(wave->samples),
      sizeof(float) * wave->num_samples,
      [](Napi::Env /*env*/, void * /*data*/, const SherpaOnnxWave *hint) {
        SherpaOnnxFreeWave(hint);
      },
      wave);
  Napi::Float32Array float32Array =
      Napi::Float32Array::New(env, wave->num_samples, arrayBuffer, 0);

  Napi::Object obj = Napi::Object::New(env);
  obj.Set(Napi::String::New(env, "samples"), float32Array);
  obj.Set(Napi::String::New(env, "sampleRate"), wave->sample_rate);
  return obj;
}

void InitWaveReader(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "readWave"),
              Napi::Function::New(env, ReadWaveWrapper));
}
