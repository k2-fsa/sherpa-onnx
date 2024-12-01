// scripts/node-addon-api/src/wave-reader.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <algorithm>
#include <sstream>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static Napi::Object ReadWaveWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() > 2) {
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

  std::string filename = info[0].As<Napi::String>().Utf8Value();

  bool enable_external_buffer = true;
  if (info.Length() == 2) {
    if (info[1].IsBoolean()) {
      enable_external_buffer = info[1].As<Napi::Boolean>().Value();
    } else {
      Napi::TypeError::New(env, "Argument 1 should be a boolean")
          .ThrowAsJavaScriptException();

      return {};
    }
  }

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(filename.c_str());
  if (!wave) {
    std::ostringstream os;
    os << "Failed to read '" << filename << "'";
    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (enable_external_buffer) {
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
  } else {
    // don't use external buffer
    Napi::ArrayBuffer arrayBuffer =
        Napi::ArrayBuffer::New(env, sizeof(float) * wave->num_samples);

    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, wave->num_samples, arrayBuffer, 0);

    std::copy(wave->samples, wave->samples + wave->num_samples,
              float32Array.Data());

    Napi::Object obj = Napi::Object::New(env);
    obj.Set(Napi::String::New(env, "samples"), float32Array);
    obj.Set(Napi::String::New(env, "sampleRate"), wave->sample_rate);

    SherpaOnnxFreeWave(wave);

    return obj;
  }
}

static Napi::Object ReadWaveFromBinaryWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() > 2) {
    std::ostringstream os;
    os << "Expect only 1 or 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsTypedArray()) {
    Napi::TypeError::New(env, "Argument 0 should be a float32 array")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Uint8Array data = info[0].As<Napi::Uint8Array>();
  int32_t n = data.ElementLength();
  const SherpaOnnxWave *wave = SherpaOnnxReadWaveFromBinaryData(
      reinterpret_cast<const char *>(data.Data()), n);
  if (!wave) {
    std::ostringstream os;
    os << "Failed to read wave";
    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  bool enable_external_buffer = true;
  if (info.Length() == 2) {
    if (info[1].IsBoolean()) {
      enable_external_buffer = info[1].As<Napi::Boolean>().Value();
    } else {
      Napi::TypeError::New(env, "Argument 1 should be a boolean")
          .ThrowAsJavaScriptException();

      return {};
    }
  }

  if (enable_external_buffer) {
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
  } else {
    // don't use external buffer
    Napi::ArrayBuffer arrayBuffer =
        Napi::ArrayBuffer::New(env, sizeof(float) * wave->num_samples);

    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, wave->num_samples, arrayBuffer, 0);

    std::copy(wave->samples, wave->samples + wave->num_samples,
              float32Array.Data());

    Napi::Object obj = Napi::Object::New(env);
    obj.Set(Napi::String::New(env, "samples"), float32Array);
    obj.Set(Napi::String::New(env, "sampleRate"), wave->sample_rate);

    SherpaOnnxFreeWave(wave);

    return obj;
  }
}

void InitWaveReader(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "readWave"),
              Napi::Function::New(env, ReadWaveWrapper));

  exports.Set(Napi::String::New(env, "readWaveFromBinary"),
              Napi::Function::New(env, ReadWaveFromBinaryWrapper));
}
