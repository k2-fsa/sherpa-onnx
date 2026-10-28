// scripts/node-addon-api/src/offline-source-separation.cc
//
// Copyright (c)  2026  Nexmoe

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static const napi_type_tag kSourceSeparationTypeTag = {0x9d3b5ef1f2264ad1,
                                                       0x82216b13b4d8f534};

static SherpaOnnxOfflineSourceSeparationModelConfig GetModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineSourceSeparationModelConfig c{};
  if (!obj.Has("model") || !obj.Get("model").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("model").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_INT32(num_threads, numThreads);
  SHERPA_ONNX_ASSIGN_ATTR_STR(provider, provider);
  if (o.Has("debug")) {
    if (o.Get("debug").IsBoolean()) {
      c.debug = o.Get("debug").As<Napi::Boolean>().Value();
    } else if (o.Get("debug").IsNumber()) {
      c.debug = o.Get("debug").As<Napi::Number>().Int32Value();
    }
  }

  if (o.Has("spleeter") && o.Get("spleeter").IsObject()) {
    Napi::Object model = o;
    o = model.Get("spleeter").As<Napi::Object>();
    SHERPA_ONNX_ASSIGN_ATTR_STR(spleeter.vocals, vocals);
    SHERPA_ONNX_ASSIGN_ATTR_STR(spleeter.accompaniment, accompaniment);
    o = model;
  }

  if (o.Has("uvr") && o.Get("uvr").IsObject()) {
    o = o.Get("uvr").As<Napi::Object>();
    SHERPA_ONNX_ASSIGN_ATTR_STR(uvr.model, model);
  }
  return c;
}

static Napi::External<SherpaOnnxOfflineSourceSeparation>
CreateOfflineSourceSeparationWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsObject() || info[0].IsArray()) {
    Napi::TypeError::New(env, "Expect one source separation config object")
        .ThrowAsJavaScriptException();
    return {};
  }

  SherpaOnnxOfflineSourceSeparationConfig c{};
  c.model = GetModelConfig(info[0].As<Napi::Object>());
  const auto *ss = SherpaOnnxCreateOfflineSourceSeparation(&c);
  SHERPA_ONNX_DELETE_C_STR(c.model.spleeter.vocals);
  SHERPA_ONNX_DELETE_C_STR(c.model.spleeter.accompaniment);
  SHERPA_ONNX_DELETE_C_STR(c.model.uvr.model);
  SHERPA_ONNX_DELETE_C_STR(c.model.provider);

  if (!ss) {
    Napi::TypeError::New(env, "Please check your source separation config!")
        .ThrowAsJavaScriptException();
    return {};
  }

  auto handle = Napi::External<SherpaOnnxOfflineSourceSeparation>::New(
      env, const_cast<SherpaOnnxOfflineSourceSeparation *>(ss),
      [](Napi::Env, SherpaOnnxOfflineSourceSeparation *p) {
        SherpaOnnxDestroyOfflineSourceSeparation(p);
      });
  handle.TypeTag(&kSourceSeparationTypeTag);
  return handle;
}

static const SherpaOnnxOfflineSourceSeparation *GetSourceSeparation(
    const Napi::CallbackInfo &info, size_t num_args) {
  if (info.Length() != num_args || !info[0].IsExternal()) {
    Napi::TypeError::New(
        info.Env(), "Expect a source separation handle and valid arguments")
        .ThrowAsJavaScriptException();
    return nullptr;
  }
  auto handle = info[0].As<Napi::External<SherpaOnnxOfflineSourceSeparation>>();
  if (!handle.CheckTypeTag(&kSourceSeparationTypeTag)) {
    Napi::TypeError::New(info.Env(), "Expect a source separation handle")
        .ThrowAsJavaScriptException();
    return nullptr;
  }
  return handle.Data();
}

static Napi::Number GetOutputSampleRateWrapper(const Napi::CallbackInfo &info) {
  const auto *ss = GetSourceSeparation(info, 1);
  if (!ss) return {};
  return Napi::Number::New(
      info.Env(), SherpaOnnxOfflineSourceSeparationGetOutputSampleRate(ss));
}

static Napi::Number GetNumberOfStemsWrapper(const Napi::CallbackInfo &info) {
  const auto *ss = GetSourceSeparation(info, 1);
  if (!ss) return {};
  return Napi::Number::New(
      info.Env(), SherpaOnnxOfflineSourceSeparationGetNumberOfStems(ss));
}

static Napi::Object CreateOutputObject(
    Napi::Env env,
    const std::shared_ptr<const SherpaOnnxSourceSeparationOutput> &output,
    bool enable_external_buffer) {
  Napi::Array stems = Napi::Array::New(env, output->num_stems);
  for (int32_t s = 0; s != output->num_stems; ++s) {
    const auto &stem = output->stems[s];
    Napi::Array channels = Napi::Array::New(env, stem.num_channels);
    for (int32_t c = 0; c != stem.num_channels; ++c) {
      Napi::ArrayBuffer buffer;
      if (enable_external_buffer) {
        // Every channel shares ownership of the complete native output.
        buffer =
            Napi::ArrayBuffer::New(env, stem.samples[c], sizeof(float) * stem.n,
                                   [output](Napi::Env, void *) {});
      } else {
        buffer = Napi::ArrayBuffer::New(env, sizeof(float) * stem.n);
        std::copy(stem.samples[c], stem.samples[c] + stem.n,
                  static_cast<float *>(buffer.Data()));
      }
      channels.Set(c, Napi::Float32Array::New(env, stem.n, buffer, 0));
    }
    stems.Set(s, channels);
  }

  Napi::Object ans = Napi::Object::New(env);
  ans.Set("stems", stems);
  ans.Set("sampleRate", output->sample_rate);
  return ans;
}

static Napi::Object ProcessWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const auto *ss = GetSourceSeparation(info, 2);
  if (!ss) return {};

  if (!info[1].IsObject() || info[1].IsArray()) {
    Napi::TypeError::New(env, "Expect an audio request object")
        .ThrowAsJavaScriptException();
    return {};
  }
  Napi::Object obj = info[1].As<Napi::Object>();
  if (!obj.Get("samples").IsArray()) {
    Napi::TypeError::New(env,
                         "samples must be an array of Float32Array channels")
        .ThrowAsJavaScriptException();
    return {};
  }
  Napi::Array channels = obj.Get("samples").As<Napi::Array>();
  uint32_t num_channels = channels.Length();
  if (num_channels < 1 || num_channels > 2) {
    Napi::RangeError::New(env, "Expect one or two input channels")
        .ThrowAsJavaScriptException();
    return {};
  }

  if (!obj.Get("sampleRate").IsNumber()) {
    Napi::TypeError::New(env, "sampleRate must be a positive integer")
        .ThrowAsJavaScriptException();
    return {};
  }
  double sample_rate = obj.Get("sampleRate").As<Napi::Number>().DoubleValue();
  if (!std::isfinite(sample_rate) || sample_rate <= 0 ||
      sample_rate > std::numeric_limits<int32_t>::max() ||
      sample_rate != std::floor(sample_rate)) {
    Napi::RangeError::New(env, "sampleRate must be a positive int32")
        .ThrowAsJavaScriptException();
    return {};
  }

  bool enable_external_buffer = true;
  if (obj.Has("enableExternalBuffer")) {
    if (!obj.Get("enableExternalBuffer").IsBoolean()) {
      Napi::TypeError::New(env, "enableExternalBuffer must be a boolean")
          .ThrowAsJavaScriptException();
      return {};
    }
    enable_external_buffer =
        obj.Get("enableExternalBuffer").As<Napi::Boolean>().Value();
  }

  std::vector<Napi::Value> channel_values;
  for (uint32_t c = 0; c != num_channels; ++c) {
    auto channel = channels.Get(c);
    if (!channel.IsTypedArray() ||
        channel.As<Napi::TypedArray>().TypedArrayType() != napi_float32_array) {
      Napi::TypeError::New(env, "Each channel must be a Float32Array")
          .ThrowAsJavaScriptException();
      return {};
    }
    channel_values.push_back(channel);
  }

  // Read pointers after all property getters have run: a later channel getter
  // may detach an earlier channel's ArrayBuffer.
  std::vector<const float *> samples;
  size_t n = 0;
  for (const auto &channel : channel_values) {
    auto array = channel.As<Napi::Float32Array>();
    size_t length = array.ElementLength();
    if (length == 0 || length > std::numeric_limits<int32_t>::max() ||
        (!samples.empty() && length != n)) {
      Napi::RangeError::New(env,
                            "Channels must have equal, nonzero int32 lengths")
          .ThrowAsJavaScriptException();
      return {};
    }
    n = length;
    samples.push_back(array.Data());
  }
  // Spleeter expects stereo. Duplicate mono input without copying samples.
  if (samples.size() == 1) samples.push_back(samples[0]);

  try {
    std::shared_ptr<const SherpaOnnxSourceSeparationOutput> output(
        SherpaOnnxOfflineSourceSeparationProcess(
            ss, samples.data(), samples.size(), static_cast<int32_t>(n),
            static_cast<int32_t>(sample_rate)),
        SherpaOnnxDestroySourceSeparationOutput);
    if (!output) {
      Napi::Error::New(env, "Source separation failed")
          .ThrowAsJavaScriptException();
      return {};
    }
    return CreateOutputObject(env, output, enable_external_buffer);
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return {};
  }
}

void InitOfflineSourceSeparation(Napi::Env env, Napi::Object exports) {
  exports.Set("createOfflineSourceSeparation",
              Napi::Function::New(env, CreateOfflineSourceSeparationWrapper));
  exports.Set("offlineSourceSeparationGetOutputSampleRate",
              Napi::Function::New(env, GetOutputSampleRateWrapper));
  exports.Set("offlineSourceSeparationGetNumberOfStems",
              Napi::Function::New(env, GetNumberOfStemsWrapper));
  exports.Set("offlineSourceSeparationProcess",
              Napi::Function::New(env, ProcessWrapper));
}
