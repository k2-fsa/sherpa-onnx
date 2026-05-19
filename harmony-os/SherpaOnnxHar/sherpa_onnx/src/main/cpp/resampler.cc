// scripts/node-addon-api/src/resampler.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

// createLinearResampler(inputSampleRate, outputSampleRate)
// Returns an External handle to a SherpaOnnxLinearResampler.
static Napi::External<SherpaOnnxLinearResampler> CreateLinearResamplerWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect 2 arguments (inputSampleRate, outputSampleRate). Given: "
       << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsNumber()) {
    Napi::TypeError::New(env, "Argument 0 should be a number (inputSampleRate)")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsNumber()) {
    Napi::TypeError::New(env,
                         "Argument 1 should be a number (outputSampleRate)")
        .ThrowAsJavaScriptException();

    return {};
  }

  int32_t input_sample_rate = info[0].As<Napi::Number>().Int32Value();
  int32_t output_sample_rate = info[1].As<Napi::Number>().Int32Value();

  if (input_sample_rate <= 0 || output_sample_rate <= 0) {
    Napi::TypeError::New(env,
                         "inputSampleRate and outputSampleRate must be > 0")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxLinearResampler *p = SherpaOnnxCreateLinearResampler(
      input_sample_rate, output_sample_rate, 0, 0);

  return Napi::External<SherpaOnnxLinearResampler>::New(
      env, const_cast<SherpaOnnxLinearResampler *>(p),
      [](Napi::Env env, SherpaOnnxLinearResampler *ptr) {
        SherpaOnnxDestroyLinearResampler(ptr);
      });
}

// resampleLinear(resamplerHandle, samples, flush)
// Returns a Float32Array of resampled samples.
// flush should be 1 for the final chunk; 0 otherwise.
static Napi::Float32Array ResampleLinearWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 3) {
    std::ostringstream os;
    os << "Expect 3 arguments (resampler, samples, flush). Given: "
       << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a resampler handle.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsTypedArray() ||
      info[1].As<Napi::TypedArray>().TypedArrayType() != napi_float32_array) {
    Napi::TypeError::New(env, "Argument 1 should be a Float32Array.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[2].IsNumber()) {
    Napi::TypeError::New(env, "Argument 2 should be a number (flush: 0 or 1).")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxLinearResampler *p =
      info[0].As<Napi::External<SherpaOnnxLinearResampler>>().Data();

  Napi::Float32Array samples = info[1].As<Napi::Float32Array>();
  int32_t input_dim = samples.ElementLength();
  int32_t flush = info[2].As<Napi::Number>().Int32Value();

  const SherpaOnnxResampleOut *out =
      SherpaOnnxLinearResamplerResample(p, samples.Data(), input_dim, flush);

  Napi::Float32Array result = Napi::Float32Array::New(env, out->n);
  std::copy(out->samples, out->samples + out->n, result.Data());

  SherpaOnnxLinearResamplerResampleFree(out);

  return result;
}

// linearResamplerReset(resamplerHandle)
static void LinearResamplerResetWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect 1 argument (resampler). Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a resampler handle.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxLinearResampler *p =
      info[0].As<Napi::External<SherpaOnnxLinearResampler>>().Data();

  SherpaOnnxLinearResamplerReset(p);
}

// linearResamplerGetInputSampleRate(resamplerHandle)
static Napi::Number LinearResamplerGetInputSampleRateWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect 1 argument (resampler). Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a resampler handle.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxLinearResampler *p =
      info[0].As<Napi::External<SherpaOnnxLinearResampler>>().Data();

  return Napi::Number::New(
      env, SherpaOnnxLinearResamplerResampleGetInputSampleRate(p));
}

// linearResamplerGetOutputSampleRate(resamplerHandle)
static Napi::Number LinearResamplerGetOutputSampleRateWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect 1 argument (resampler). Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a resampler handle.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxLinearResampler *p =
      info[0].As<Napi::External<SherpaOnnxLinearResampler>>().Data();

  return Napi::Number::New(
      env, SherpaOnnxLinearResamplerResampleGetOutputSampleRate(p));
}

void InitResampler(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createLinearResampler"),
              Napi::Function::New(env, CreateLinearResamplerWrapper));

  exports.Set(Napi::String::New(env, "resampleLinear"),
              Napi::Function::New(env, ResampleLinearWrapper));

  exports.Set(Napi::String::New(env, "linearResamplerReset"),
              Napi::Function::New(env, LinearResamplerResetWrapper));

  exports.Set(Napi::String::New(env, "linearResamplerGetInputSampleRate"),
              Napi::Function::New(env, LinearResamplerGetInputSampleRateWrapper));

  exports.Set(
      Napi::String::New(env, "linearResamplerGetOutputSampleRate"),
      Napi::Function::New(env, LinearResamplerGetOutputSampleRateWrapper));
}
