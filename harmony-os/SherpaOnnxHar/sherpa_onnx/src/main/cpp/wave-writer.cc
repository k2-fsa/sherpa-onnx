// scripts/node-addon-api/src/wave-writer.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

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
#if __OHOS__
  int32_t ok = SherpaOnnxWriteWave(
      samples.Data(), samples.ElementLength() / sizeof(float), sample_rate,
      info[0].As<Napi::String>().Utf8Value().c_str());
#else
  int32_t ok =
      SherpaOnnxWriteWave(samples.Data(), samples.ElementLength(), sample_rate,
                          info[0].As<Napi::String>().Utf8Value().c_str());
#endif

  return Napi::Boolean::New(env, ok);
}

// (filename, {samples: [samples0, samples1, ...], sampleRate: sampleRate}
static Napi::Boolean WriteWaveMultiChannelWrapper(
    const Napi::CallbackInfo &info) {
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

  if (!obj.Has("samples") || !obj.Get("samples").IsArray()) {
    Napi::TypeError::New(env, "The object['samples'] should be an array")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("sampleRate") || !obj.Get("sampleRate").IsNumber()) {
    Napi::TypeError::New(env, "The object['sampleRate'] should be a number")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Array channels = obj.Get("samples").As<Napi::Array>();
  uint32_t num_channels = channels.Length();
  if (num_channels == 0) {
    Napi::RangeError::New(env, "The object['samples'] should not be empty")
        .ThrowAsJavaScriptException();

    return {};
  }

  double sample_rate = obj.Get("sampleRate").As<Napi::Number>().DoubleValue();
  if (!std::isfinite(sample_rate) || sample_rate <= 0 ||
      sample_rate > std::numeric_limits<int32_t>::max() ||
      sample_rate != std::floor(sample_rate)) {
    Napi::RangeError::New(
        env, "The object['sampleRate'] should be a positive integer")
        .ThrowAsJavaScriptException();

    return {};
  }

  // Read pointers after all property getters have run: a later channel getter
  // may detach an earlier channel's ArrayBuffer.
  std::vector<Napi::Value> channel_values;
  for (uint32_t c = 0; c != num_channels; ++c) {
    Napi::Value value = channels.Get(c);
    if (!value.IsTypedArray() ||
        value.As<Napi::TypedArray>().TypedArrayType() != napi_float32_array) {
      Napi::TypeError::New(env, "Each channel should be a Float32Array")
          .ThrowAsJavaScriptException();

      return {};
    }
    channel_values.push_back(value);
  }

  std::vector<const float *> channel_ptrs;
  int32_t n = 0;
  for (const auto &value : channel_values) {
    Napi::Float32Array samples = value.As<Napi::Float32Array>();
    int32_t length = samples.ElementLength();
#if __OHOS__
    length /= sizeof(float);
#endif
    if (length <= 0 || (n != 0 && length != n)) {
      Napi::RangeError::New(env, "Channels should have equal, nonzero lengths")
          .ThrowAsJavaScriptException();

      return {};
    }
    n = length;
    channel_ptrs.push_back(samples.Data());
  }

  int32_t ok = SherpaOnnxWriteWaveMultiChannel(
      channel_ptrs.data(), n, static_cast<int32_t>(sample_rate), num_channels,
      info[0].As<Napi::String>().Utf8Value().c_str());

  return Napi::Boolean::New(env, ok);
}

void InitWaveWriter(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "writeWave"),
              Napi::Function::New(env, WriteWaveWrapper));

  exports.Set(Napi::String::New(env, "writeWaveMultiChannel"),
              Napi::Function::New(env, WriteWaveMultiChannelWrapper));
}
