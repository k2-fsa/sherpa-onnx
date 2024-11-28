// scripts/node-addon-api/src/audio-tagging.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxOfflineZipformerAudioTaggingModelConfig
GetAudioTaggingZipformerModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineZipformerAudioTaggingModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("zipformer") || !obj.Get("zipformer").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("zipformer").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);

  return c;
}

static SherpaOnnxAudioTaggingModelConfig GetAudioTaggingModelConfig(
    Napi::Object obj) {
  SherpaOnnxAudioTaggingModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("model") || !obj.Get("model").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("model").As<Napi::Object>();
  c.zipformer = GetAudioTaggingZipformerModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_STR(ced, ced);

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

static Napi::External<SherpaOnnxAudioTagging> CreateAudioTaggingWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsObject()) {
    Napi::TypeError::New(env, "You should pass an object as the only argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Object o = info[0].As<Napi::Object>();

  SherpaOnnxAudioTaggingConfig c;
  memset(&c, 0, sizeof(c));
  c.model = GetAudioTaggingModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_STR(labels, labels);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(top_k, topK);

  const SherpaOnnxAudioTagging *at = SherpaOnnxCreateAudioTagging(&c);

  if (c.model.zipformer.model) {
    delete[] c.model.zipformer.model;
  }

  if (c.model.ced) {
    delete[] c.model.ced;
  }

  if (c.model.provider) {
    delete[] c.model.provider;
  }

  if (c.labels) {
    delete[] c.labels;
  }

  if (!at) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxAudioTagging>::New(
      env, const_cast<SherpaOnnxAudioTagging *>(at),
      [](Napi::Env env, SherpaOnnxAudioTagging *at) {
        SherpaOnnxDestroyAudioTagging(at);
      });
}

static Napi::External<SherpaOnnxOfflineStream>
AudioTaggingCreateOfflineStreamWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(
        env, "You should pass an audio tagging pointer as the only argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxAudioTagging *at =
      info[0].As<Napi::External<SherpaOnnxAudioTagging>>().Data();

  const SherpaOnnxOfflineStream *stream =
      SherpaOnnxAudioTaggingCreateOfflineStream(at);

  return Napi::External<SherpaOnnxOfflineStream>::New(
      env, const_cast<SherpaOnnxOfflineStream *>(stream),
      [](Napi::Env env, SherpaOnnxOfflineStream *stream) {
        SherpaOnnxDestroyOfflineStream(stream);
      });
}

static Napi::Object AudioTaggingComputeWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3) {
    std::ostringstream os;
    os << "Expect only 3 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(
        env, "You should pass an audio tagging pointer as the first argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(
        env, "You should pass an offline stream pointer as the second argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[2].IsNumber()) {
    Napi::TypeError::New(env,
                         "You should pass an integer as the third argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxAudioTagging *at =
      info[0].As<Napi::External<SherpaOnnxAudioTagging>>().Data();

  SherpaOnnxOfflineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOfflineStream>>().Data();

  int32_t top_k = info[2].As<Napi::Number>().Int32Value();

  const SherpaOnnxAudioEvent *const *events =
      SherpaOnnxAudioTaggingCompute(at, stream, top_k);

  auto p = events;
  int32_t k = 0;
  while (p && *p) {
    ++k;
    ++p;
  }

  Napi::Array ans = Napi::Array::New(env, k);
  for (uint32_t i = 0; i != k; ++i) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set(Napi::String::New(env, "name"),
            Napi::String::New(env, events[i]->name));
    obj.Set(Napi::String::New(env, "index"),
            Napi::Number::New(env, events[i]->index));
    obj.Set(Napi::String::New(env, "prob"),
            Napi::Number::New(env, events[i]->prob));
    ans[i] = obj;
  }

  SherpaOnnxAudioTaggingFreeResults(events);

  return ans;
}

void InitAudioTagging(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createAudioTagging"),
              Napi::Function::New(env, CreateAudioTaggingWrapper));

  exports.Set(Napi::String::New(env, "audioTaggingCreateOfflineStream"),
              Napi::Function::New(env, AudioTaggingCreateOfflineStreamWrapper));

  exports.Set(Napi::String::New(env, "audioTaggingCompute"),
              Napi::Function::New(env, AudioTaggingComputeWrapper));
}
