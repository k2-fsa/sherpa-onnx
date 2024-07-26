// scripts/node-addon-api/src/speaker-identification.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <algorithm>
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static Napi::External<SherpaOnnxSpeakerEmbeddingExtractor>
CreateSpeakerEmbeddingExtractorWrapper(const Napi::CallbackInfo &info) {
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

  SherpaOnnxSpeakerEmbeddingExtractorConfig c;
  memset(&c, 0, sizeof(c));

  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
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

  const SherpaOnnxSpeakerEmbeddingExtractor *extractor =
      SherpaOnnxCreateSpeakerEmbeddingExtractor(&c);

  if (c.model) {
    delete[] c.model;
  }

  if (c.provider) {
    delete[] c.provider;
  }

  if (!extractor) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxSpeakerEmbeddingExtractor>::New(
      env, const_cast<SherpaOnnxSpeakerEmbeddingExtractor *>(extractor),
      [](Napi::Env env, SherpaOnnxSpeakerEmbeddingExtractor *extractor) {
        SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor);
      });
}

static Napi::Number SpeakerEmbeddingExtractorDimWrapper(
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
        env, "Argument 0 should be a speaker embedding extractor pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingExtractor *extractor =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingExtractor>>().Data();

  int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor);

  return Napi::Number::New(env, dim);
}

static Napi::External<SherpaOnnxOnlineStream>
SpeakerEmbeddingExtractorCreateStreamWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "You should pass a speaker embedding extractor "
                         "pointer as the only argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingExtractor *extractor =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingExtractor>>().Data();

  const SherpaOnnxOnlineStream *stream =
      SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor);

  return Napi::External<SherpaOnnxOnlineStream>::New(
      env, const_cast<SherpaOnnxOnlineStream *>(stream),
      [](Napi::Env env, SherpaOnnxOnlineStream *stream) {
        SherpaOnnxDestroyOnlineStream(stream);
      });
}

static Napi::Boolean SpeakerEmbeddingExtractorIsReadyWrapper(
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
        env, "Argument 0 should be a speaker embedding extractor pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingExtractor *extractor =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingExtractor>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  int32_t is_ready =
      SherpaOnnxSpeakerEmbeddingExtractorIsReady(extractor, stream);

  return Napi::Boolean::New(env, is_ready);
}

static Napi::Float32Array SpeakerEmbeddingExtractorComputeEmbeddingWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 && info.Length() != 3) {
    std::ostringstream os;
    os << "Expect only 2 or 3 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(
        env, "Argument 0 should be a speaker embedding extractor pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  bool enable_external_buffer = true;
  if (info.Length() == 3) {
    if (info[2].IsBoolean()) {
      enable_external_buffer = info[2].As<Napi::Boolean>().Value();
    } else {
      Napi::TypeError::New(env, "Argument 2 should be a boolean.")
          .ThrowAsJavaScriptException();
    }
  }

  SherpaOnnxSpeakerEmbeddingExtractor *extractor =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingExtractor>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  const float *v =
      SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor, stream);

  int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor);

  if (enable_external_buffer) {
    Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
        env, const_cast<float *>(v), sizeof(float) * dim,
        [](Napi::Env /*env*/, void *data) {
          SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(
              reinterpret_cast<float *>(data));
        });

    return Napi::Float32Array::New(env, dim, arrayBuffer, 0);
  } else {
    // don't use external buffer
    Napi::ArrayBuffer arrayBuffer =
        Napi::ArrayBuffer::New(env, sizeof(float) * dim);

    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, dim, arrayBuffer, 0);

    std::copy(v, v + dim, float32Array.Data());

    SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(v);

    return float32Array;
  }
}

static Napi::External<SherpaOnnxSpeakerEmbeddingManager>
CreateSpeakerEmbeddingManagerWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsNumber()) {
    Napi::TypeError::New(env,
                         "You should pass an integer as the only argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  int32_t dim = info[0].As<Napi::Number>().Int32Value();

  const SherpaOnnxSpeakerEmbeddingManager *manager =
      SherpaOnnxCreateSpeakerEmbeddingManager(dim);

  if (!manager) {
    Napi::TypeError::New(env, "Please check your input dim!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxSpeakerEmbeddingManager>::New(
      env, const_cast<SherpaOnnxSpeakerEmbeddingManager *>(manager),
      [](Napi::Env env, SherpaOnnxSpeakerEmbeddingManager *manager) {
        SherpaOnnxDestroySpeakerEmbeddingManager(manager);
      });
}

static Napi::Boolean SpeakerEmbeddingManagerAddWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "You should pass a speaker embedding manager pointer "
                         "as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Argument 1 should be an object")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingManager *manager =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingManager>>().Data();

  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("v")) {
    Napi::TypeError::New(env, "The argument object should have a field v")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("v").IsTypedArray()) {
    Napi::TypeError::New(env, "The object['v'] should be a typed array")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("name")) {
    Napi::TypeError::New(env, "The argument object should have a field name")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("name").IsString()) {
    Napi::TypeError::New(env, "The object['name'] should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Float32Array v = obj.Get("v").As<Napi::Float32Array>();
  Napi::String js_name = obj.Get("name").As<Napi::String>();
  std::string name = js_name.Utf8Value();

  int32_t ok =
      SherpaOnnxSpeakerEmbeddingManagerAdd(manager, name.c_str(), v.Data());
  return Napi::Boolean::New(env, ok);
}

static Napi::Boolean SpeakerEmbeddingManagerAddListFlattenedWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "You should pass a speaker embedding manager pointer "
                         "as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Argument 1 should be an object")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingManager *manager =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingManager>>().Data();

  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("vv")) {
    Napi::TypeError::New(env, "The argument object should have a field vv")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("vv").IsTypedArray()) {
    Napi::TypeError::New(env, "The object['vv'] should be a typed array")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("name")) {
    Napi::TypeError::New(env, "The argument object should have a field name")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("name").IsString()) {
    Napi::TypeError::New(env, "The object['name'] should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("n")) {
    Napi::TypeError::New(env, "The argument object should have a field n")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("n").IsNumber()) {
    Napi::TypeError::New(env, "The object['n'] should be an integer")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Float32Array v = obj.Get("vv").As<Napi::Float32Array>();
  Napi::String js_name = obj.Get("name").As<Napi::String>();
  int32_t n = obj.Get("n").As<Napi::Number>().Int32Value();

  std::string name = js_name.Utf8Value();

  int32_t ok = SherpaOnnxSpeakerEmbeddingManagerAddListFlattened(
      manager, name.c_str(), v.Data(), n);

  return Napi::Boolean::New(env, ok);
}

static Napi::Boolean SpeakerEmbeddingManagerRemoveWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "You should pass a speaker embedding manager pointer "
                         "as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsString()) {
    Napi::TypeError::New(env, "Argument 1 should be string")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingManager *manager =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingManager>>().Data();

  Napi::String js_name = info[1].As<Napi::String>();
  std::string name = js_name.Utf8Value();

  int32_t ok = SherpaOnnxSpeakerEmbeddingManagerRemove(manager, name.c_str());

  return Napi::Boolean::New(env, ok);
}

static Napi::String SpeakerEmbeddingManagerSearchWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "You should pass a speaker embedding manager pointer "
                         "as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Argument 1 should be an object")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingManager *manager =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingManager>>().Data();

  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("v")) {
    Napi::TypeError::New(env, "The argument object should have a field v")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("v").IsTypedArray()) {
    Napi::TypeError::New(env, "The object['v'] should be a typed array")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("threshold")) {
    Napi::TypeError::New(env,
                         "The argument object should have a field threshold")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("threshold").IsNumber()) {
    Napi::TypeError::New(env, "The object['threshold'] should be a float")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Float32Array v = obj.Get("v").As<Napi::Float32Array>();
  float threshold = obj.Get("threshold").As<Napi::Number>().FloatValue();

  const char *name =
      SherpaOnnxSpeakerEmbeddingManagerSearch(manager, v.Data(), threshold);
  const char *p = name;
  if (!p) {
    p = "";
  }

  Napi::String js_name = Napi::String::New(env, p);
  SherpaOnnxSpeakerEmbeddingManagerFreeSearch(name);

  return js_name;
}

static Napi::Boolean SpeakerEmbeddingManagerVerifyWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "You should pass a speaker embedding manager pointer "
                         "as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Argument 1 should be an object")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingManager *manager =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingManager>>().Data();

  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("v")) {
    Napi::TypeError::New(env, "The argument object should have a field v")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("v").IsTypedArray()) {
    Napi::TypeError::New(env, "The object['v'] should be a typed array")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("threshold")) {
    Napi::TypeError::New(env,
                         "The argument object should have a field threshold")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("threshold").IsNumber()) {
    Napi::TypeError::New(env, "The object['threshold'] should be a float")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("name")) {
    Napi::TypeError::New(env, "The argument object should have a field name")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("name").IsString()) {
    Napi::TypeError::New(env, "The object['name'] should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Float32Array v = obj.Get("v").As<Napi::Float32Array>();
  float threshold = obj.Get("threshold").As<Napi::Number>().FloatValue();

  Napi::String js_name = obj.Get("name").As<Napi::String>();
  std::string name = js_name.Utf8Value();

  int32_t found = SherpaOnnxSpeakerEmbeddingManagerVerify(manager, name.c_str(),
                                                          v.Data(), threshold);

  return Napi::Boolean::New(env, found);
}

static Napi::Boolean SpeakerEmbeddingManagerContainsWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "You should pass a speaker embedding manager pointer "
                         "as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsString()) {
    Napi::TypeError::New(env, "Argument 1 should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingManager *manager =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingManager>>().Data();

  Napi::String js_name = info[1].As<Napi::String>();
  std::string name = js_name.Utf8Value();

  int32_t exists =
      SherpaOnnxSpeakerEmbeddingManagerContains(manager, name.c_str());

  return Napi::Boolean::New(env, exists);
}

static Napi::Number SpeakerEmbeddingManagerNumSpeakersWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "You should pass a speaker embedding manager pointer "
                         "as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingManager *manager =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingManager>>().Data();

  int32_t num_speakers = SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(manager);

  return Napi::Number::New(env, num_speakers);
}

static Napi::Array SpeakerEmbeddingManagerGetAllSpeakersWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "You should pass a speaker embedding manager pointer "
                         "as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpeakerEmbeddingManager *manager =
      info[0].As<Napi::External<SherpaOnnxSpeakerEmbeddingManager>>().Data();

  int32_t num_speakers = SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(manager);
  if (num_speakers == 0) {
    return {};
  }

  const char *const *all_speaker_names =
      SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(manager);

  Napi::Array ans = Napi::Array::New(env, num_speakers);
  for (uint32_t i = 0; i != num_speakers; ++i) {
    ans[i] = Napi::String::New(env, all_speaker_names[i]);
  }
  SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(all_speaker_names);
  return ans;
}

void InitSpeakerID(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createSpeakerEmbeddingExtractor"),
              Napi::Function::New(env, CreateSpeakerEmbeddingExtractorWrapper));

  exports.Set(Napi::String::New(env, "speakerEmbeddingExtractorDim"),
              Napi::Function::New(env, SpeakerEmbeddingExtractorDimWrapper));

  exports.Set(
      Napi::String::New(env, "speakerEmbeddingExtractorCreateStream"),
      Napi::Function::New(env, SpeakerEmbeddingExtractorCreateStreamWrapper));

  exports.Set(
      Napi::String::New(env, "speakerEmbeddingExtractorIsReady"),
      Napi::Function::New(env, SpeakerEmbeddingExtractorIsReadyWrapper));

  exports.Set(
      Napi::String::New(env, "speakerEmbeddingExtractorComputeEmbedding"),
      Napi::Function::New(env,
                          SpeakerEmbeddingExtractorComputeEmbeddingWrapper));

  exports.Set(Napi::String::New(env, "createSpeakerEmbeddingManager"),
              Napi::Function::New(env, CreateSpeakerEmbeddingManagerWrapper));

  exports.Set(Napi::String::New(env, "speakerEmbeddingManagerAdd"),
              Napi::Function::New(env, SpeakerEmbeddingManagerAddWrapper));

  exports.Set(
      Napi::String::New(env, "speakerEmbeddingManagerAddListFlattened"),
      Napi::Function::New(env, SpeakerEmbeddingManagerAddListFlattenedWrapper));

  exports.Set(Napi::String::New(env, "speakerEmbeddingManagerRemove"),
              Napi::Function::New(env, SpeakerEmbeddingManagerRemoveWrapper));

  exports.Set(Napi::String::New(env, "speakerEmbeddingManagerSearch"),
              Napi::Function::New(env, SpeakerEmbeddingManagerSearchWrapper));

  exports.Set(Napi::String::New(env, "speakerEmbeddingManagerVerify"),
              Napi::Function::New(env, SpeakerEmbeddingManagerVerifyWrapper));

  exports.Set(Napi::String::New(env, "speakerEmbeddingManagerContains"),
              Napi::Function::New(env, SpeakerEmbeddingManagerContainsWrapper));

  exports.Set(
      Napi::String::New(env, "speakerEmbeddingManagerNumSpeakers"),
      Napi::Function::New(env, SpeakerEmbeddingManagerNumSpeakersWrapper));

  exports.Set(
      Napi::String::New(env, "speakerEmbeddingManagerGetAllSpeakers"),
      Napi::Function::New(env, SpeakerEmbeddingManagerGetAllSpeakersWrapper));
}
