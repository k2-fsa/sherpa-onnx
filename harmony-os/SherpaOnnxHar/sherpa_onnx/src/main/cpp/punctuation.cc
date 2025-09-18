// scripts/node-addon-api/src/punctuation.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

// offline punctuation models
static SherpaOnnxOfflinePunctuationModelConfig GetOfflinePunctuationModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflinePunctuationModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("model") || !obj.Get("model").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("model").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(ct_transformer, ctTransformer);

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

static Napi::External<SherpaOnnxOfflinePunctuation>
CreateOfflinePunctuationWrapper(const Napi::CallbackInfo &info) {
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

  SherpaOnnxOfflinePunctuationConfig c;
  memset(&c, 0, sizeof(c));
  c.model = GetOfflinePunctuationModelConfig(o);

  const SherpaOnnxOfflinePunctuation *punct =
      SherpaOnnxCreateOfflinePunctuation(&c);

  SHERPA_ONNX_DELETE_C_STR(c.model.ct_transformer);
  SHERPA_ONNX_DELETE_C_STR(c.model.provider);

  if (!punct) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxOfflinePunctuation>::New(
      env, const_cast<SherpaOnnxOfflinePunctuation *>(punct),
      [](Napi::Env env, SherpaOnnxOfflinePunctuation *punct) {
        SherpaOnnxDestroyOfflinePunctuation(punct);
      });
}

static Napi::String OfflinePunctuationAddPunctWraper(
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
        env,
        "You should pass an offline punctuation pointer as the first argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsString()) {
    Napi::TypeError::New(env, "You should pass a string as the second argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOfflinePunctuation *punct =
      info[0].As<Napi::External<SherpaOnnxOfflinePunctuation>>().Data();
  Napi::String js_text = info[1].As<Napi::String>();
  std::string text = js_text.Utf8Value();

  const char *punct_text =
      SherpaOfflinePunctuationAddPunct(punct, text.c_str());

  Napi::String ans = Napi::String::New(env, punct_text);
  SherpaOfflinePunctuationFreeText(punct_text);
  return ans;
}

// online punctuation models
static SherpaOnnxOnlinePunctuationModelConfig GetOnlinePunctuationModelConfig(
    Napi::Object obj) {
  SherpaOnnxOnlinePunctuationModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("model") || !obj.Get("model").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("model").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(cnn_bilstm, cnnBilstm);

  SHERPA_ONNX_ASSIGN_ATTR_STR(bpe_vocab, bpeVocab);

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

static Napi::External<SherpaOnnxOnlinePunctuation>
CreateOnlinePunctuationWrapper(const Napi::CallbackInfo &info) {
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

  SherpaOnnxOnlinePunctuationConfig c;
  memset(&c, 0, sizeof(c));
  c.model = GetOnlinePunctuationModelConfig(o);

  const SherpaOnnxOnlinePunctuation *punct =
      SherpaOnnxCreateOnlinePunctuation(&c);

  SHERPA_ONNX_DELETE_C_STR(c.model.cnn_bilstm);
  SHERPA_ONNX_DELETE_C_STR(c.model.bpe_vocab);
  SHERPA_ONNX_DELETE_C_STR(c.model.provider);

  if (!punct) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxOnlinePunctuation>::New(
      env, const_cast<SherpaOnnxOnlinePunctuation *>(punct),
      [](Napi::Env env, SherpaOnnxOnlinePunctuation *punct) {
        SherpaOnnxDestroyOnlinePunctuation(punct);
      });
}

static Napi::String OnlinePunctuationAddPunctWraper(
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
        env,
        "You should pass an online punctuation pointer as the first argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsString()) {
    Napi::TypeError::New(env, "You should pass a string as the second argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOnlinePunctuation *punct =
      info[0].As<Napi::External<SherpaOnnxOnlinePunctuation>>().Data();
  Napi::String js_text = info[1].As<Napi::String>();
  std::string text = js_text.Utf8Value();

  const char *punct_text =
      SherpaOnnxOnlinePunctuationAddPunct(punct, text.c_str());

  Napi::String ans = Napi::String::New(env, punct_text);
  SherpaOnnxOnlinePunctuationFreeText(punct_text);
  return ans;
}

// exports

void InitPunctuation(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOfflinePunctuation"),
              Napi::Function::New(env, CreateOfflinePunctuationWrapper));

  exports.Set(Napi::String::New(env, "offlinePunctuationAddPunct"),
              Napi::Function::New(env, OfflinePunctuationAddPunctWraper));

  exports.Set(Napi::String::New(env, "createOnlinePunctuation"),
              Napi::Function::New(env, CreateOnlinePunctuationWrapper));

  exports.Set(Napi::String::New(env, "onlinePunctuationAddPunct"),
              Napi::Function::New(env, OnlinePunctuationAddPunctWraper));
}
