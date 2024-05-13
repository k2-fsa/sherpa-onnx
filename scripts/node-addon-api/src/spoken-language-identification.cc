// scripts/node-addon-api/src/spoken-language-identification.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <sstream>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxSpokenLanguageIdentificationWhisperConfig
GetSpokenLanguageIdentificationWhisperConfig(Napi::Object obj) {
  SherpaOnnxSpokenLanguageIdentificationWhisperConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("whisper") || !obj.Get("whisper").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("whisper").As<Napi::Object>();

  if (o.Has("encoder") && o.Get("encoder").IsString()) {
    Napi::String encoder = o.Get("encoder").As<Napi::String>();
    std::string s = encoder.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.encoder = p;
  }

  if (o.Has("decoder") && o.Get("decoder").IsString()) {
    Napi::String decoder = o.Get("decoder").As<Napi::String>();
    std::string s = decoder.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.decoder = p;
  }

  if (o.Has("tailPaddings") && o.Get("tailPaddings").IsNumber()) {
    c.tail_paddings = o.Get("tailPaddings").As<Napi::Number>().Int32Value();
  }

  return c;
}

static Napi::External<SherpaOnnxSpokenLanguageIdentification>
CreateSpokenLanguageIdentificationWrapper(const Napi::CallbackInfo &info) {
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

  SherpaOnnxSpokenLanguageIdentificationConfig c;
  memset(&c, 0, sizeof(c));
  c.whisper = GetSpokenLanguageIdentificationWhisperConfig(o);

  if (o.Has("numThreads") && o.Get("numThreads").IsNumber()) {
    c.num_threads = o.Get("numThreads").As<Napi::Number>().Int32Value();
  }

  if (o.Has("debug") &&
      (o.Get("debug").IsNumber() || o.Get("debug").IsBoolean())) {
    if (o.Get("debug").IsBoolean()) {
      c.debug = o.Get("debug").As<Napi::Boolean>().Value();
    } else {
      c.debug = o.Get("debug").As<Napi::Number>().Int32Value();
    }
  }

  if (o.Has("provider") && o.Get("provider").IsString()) {
    Napi::String provider = o.Get("provider").As<Napi::String>();
    std::string s = provider.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.provider = p;
  }

  const SherpaOnnxSpokenLanguageIdentification *slid =
      SherpaOnnxCreateSpokenLanguageIdentification(&c);

  if (c.whisper.encoder) {
    delete[] c.whisper.encoder;
  }

  if (c.whisper.decoder) {
    delete[] c.whisper.decoder;
  }

  if (c.provider) {
    delete[] c.provider;
  }

  if (!slid) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxSpokenLanguageIdentification>::New(
      env, const_cast<SherpaOnnxSpokenLanguageIdentification *>(slid),
      [](Napi::Env env, SherpaOnnxSpokenLanguageIdentification *slid) {
        SherpaOnnxDestroySpokenLanguageIdentification(slid);
      });
}

static Napi::External<SherpaOnnxOfflineStream>
SpokenLanguageIdentificationCreateOfflineStreamWrapper(
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
        env,
        "You should pass an offline language ID pointer as the only argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpokenLanguageIdentification *slid =
      info[0]
          .As<Napi::External<SherpaOnnxSpokenLanguageIdentification>>()
          .Data();

  SherpaOnnxOfflineStream *stream =
      SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(slid);

  return Napi::External<SherpaOnnxOfflineStream>::New(
      env, stream, [](Napi::Env env, SherpaOnnxOfflineStream *stream) {
        DestroyOfflineStream(stream);
      });
}

static Napi::String SpokenLanguageIdentificationComputeWrapper(
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
        env, "Argument 0 should be an offline spoken language ID pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an offline stream pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxSpokenLanguageIdentification *slid =
      info[0]
          .As<Napi::External<SherpaOnnxSpokenLanguageIdentification>>()
          .Data();

  SherpaOnnxOfflineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOfflineStream>>().Data();

  const SherpaOnnxSpokenLanguageIdentificationResult *r =
      SherpaOnnxSpokenLanguageIdentificationCompute(slid, stream);

  std::string lang = r->lang;
  SherpaOnnxDestroySpokenLanguageIdentificationResult(r);

  return Napi::String::New(env, lang);
}

void InitSpokenLanguageID(Napi::Env env, Napi::Object exports) {
  exports.Set(
      Napi::String::New(env, "createSpokenLanguageIdentification"),
      Napi::Function::New(env, CreateSpokenLanguageIdentificationWrapper));

  exports.Set(
      Napi::String::New(env, "createSpokenLanguageIdentificationOfflineStream"),
      Napi::Function::New(
          env, SpokenLanguageIdentificationCreateOfflineStreamWrapper));

  exports.Set(
      Napi::String::New(env, "spokenLanguageIdentificationCompute"),
      Napi::Function::New(env, SpokenLanguageIdentificationComputeWrapper));
}
