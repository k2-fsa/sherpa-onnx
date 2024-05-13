// scripts/node-addon-api/src/non-streaming-tts.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <sstream>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxOfflineTtsVitsModelConfig GetOfflineTtsVitsModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTtsVitsModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("vits") || !obj.Get("vits").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("vits").As<Napi::Object>();

  if (o.Has("model") && o.Get("model").IsString()) {
    Napi::String model = o.Get("model").As<Napi::String>();
    std::string s = model.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.model = p;
  }

  if (o.Has("lexicon") && o.Get("lexicon").IsString()) {
    Napi::String lexicon = o.Get("lexicon").As<Napi::String>();
    std::string s = lexicon.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.lexicon = p;
  }

  if (o.Has("tokens") && o.Get("tokens").IsString()) {
    Napi::String tokens = o.Get("tokens").As<Napi::String>();
    std::string s = tokens.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.tokens = p;
  }

  if (o.Has("dataDir") && o.Get("dataDir").IsString()) {
    Napi::String data_dir = o.Get("dataDir").As<Napi::String>();
    std::string s = data_dir.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.data_dir = p;
  }

  if (o.Has("noiseScale") && o.Get("noiseScale").IsNumber()) {
    c.noise_scale = o.Get("noiseScale").As<Napi::Number>().FloatValue();
  }

  if (o.Has("noiseScaleW") && o.Get("noiseScaleW").IsNumber()) {
    c.noise_scale_w = o.Get("noiseScaleW").As<Napi::Number>().FloatValue();
  }

  if (o.Has("lengthScale") && o.Get("lengthScale").IsNumber()) {
    c.length_scale = o.Get("lengthScale").As<Napi::Number>().FloatValue();
  }

  if (o.Has("dictDir") && o.Get("dictDir").IsString()) {
    Napi::String dict_dir = o.Get("dictDir").As<Napi::String>();
    std::string s = dict_dir.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.dict_dir = p;
  }

  return c;
}

static SherpaOnnxOfflineTtsModelConfig GetOfflineTtsModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTtsModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("model") || !obj.Get("model").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("model").As<Napi::Object>();

  c.vits = GetOfflineTtsVitsModelConfig(o);

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

  return c;
}

static Napi::External<SherpaOnnxOfflineTts> CreateOfflineTtsWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsObject()) {
    Napi::TypeError::New(env, "Expect an object as the argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Object o = info[0].As<Napi::Object>();

  SherpaOnnxOfflineTtsConfig c;
  memset(&c, 0, sizeof(c));

  c.model = GetOfflineTtsModelConfig(o);

  if (o.Has("ruleFsts") && o.Get("ruleFsts").IsString()) {
    Napi::String rule_fsts = o.Get("ruleFsts").As<Napi::String>();
    std::string s = rule_fsts.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.rule_fsts = p;
  }

  if (o.Has("maxNumSentences") && o.Get("maxNumSentences").IsNumber()) {
    c.max_num_sentences =
        o.Get("maxNumSentences").As<Napi::Number>().Int32Value();
  }

  if (o.Has("ruleFars") && o.Get("ruleFars").IsString()) {
    Napi::String rule_fars = o.Get("ruleFars").As<Napi::String>();
    std::string s = rule_fars.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.rule_fars = p;
  }

  SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&c);

  if (c.model.vits.model) {
    delete[] c.model.vits.model;
  }

  if (c.model.vits.lexicon) {
    delete[] c.model.vits.lexicon;
  }

  if (c.model.vits.tokens) {
    delete[] c.model.vits.tokens;
  }

  if (c.model.vits.data_dir) {
    delete[] c.model.vits.data_dir;
  }

  if (c.model.vits.dict_dir) {
    delete[] c.model.vits.dict_dir;
  }

  if (c.model.provider) {
    delete[] c.model.provider;
  }

  if (c.rule_fsts) {
    delete[] c.rule_fsts;
  }

  if (c.rule_fars) {
    delete[] c.rule_fars;
  }

  if (!tts) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxOfflineTts>::New(
      env, tts, [](Napi::Env env, SherpaOnnxOfflineTts *tts) {
        SherpaOnnxDestroyOfflineTts(tts);
      });
}

static Napi::Number OfflineTtsSampleRateWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxOfflineTts *tts =
      info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

  int32_t sample_rate = SherpaOnnxOfflineTtsSampleRate(tts);

  return Napi::Number::New(env, sample_rate);
}

static Napi::Number OfflineTtsNumSpeakersWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxOfflineTts *tts =
      info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

  int32_t num_speakers = SherpaOnnxOfflineTtsNumSpeakers(tts);

  return Napi::Number::New(env, num_speakers);
}

static Napi::Object OfflineTtsGenerateWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxOfflineTts *tts =
      info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Argument 1 should be an object")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("text")) {
    Napi::TypeError::New(env, "The argument object should have a field text")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("text").IsString()) {
    Napi::TypeError::New(env, "The object['text'] should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("sid")) {
    Napi::TypeError::New(env, "The argument object should have a field sid")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("sid").IsNumber()) {
    Napi::TypeError::New(env, "The object['sid'] should be a number")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("speed")) {
    Napi::TypeError::New(env, "The argument object should have a field speed")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("speed").IsNumber()) {
    Napi::TypeError::New(env, "The object['speed'] should be a number")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::String _text = obj.Get("text").As<Napi::String>();
  std::string text = _text.Utf8Value();
  int32_t sid = obj.Get("sid").As<Napi::Number>().Int32Value();
  float speed = obj.Get("speed").As<Napi::Number>().FloatValue();

  const SherpaOnnxGeneratedAudio *audio =
      SherpaOnnxOfflineTtsGenerate(tts, text.c_str(), sid, speed);

  Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
      env, const_cast<float *>(audio->samples), sizeof(float) * audio->n,
      [](Napi::Env /*env*/, void * /*data*/,
         const SherpaOnnxGeneratedAudio *hint) {
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(hint);
      },
      audio);
  Napi::Float32Array float32Array =
      Napi::Float32Array::New(env, audio->n, arrayBuffer, 0);

  Napi::Object ans = Napi::Object::New(env);
  ans.Set(Napi::String::New(env, "samples"), float32Array);
  ans.Set(Napi::String::New(env, "sampleRate"), audio->sample_rate);
  return ans;
}

void InitNonStreamingTts(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOfflineTts"),
              Napi::Function::New(env, CreateOfflineTtsWrapper));

  exports.Set(Napi::String::New(env, "getOfflineTtsSampleRate"),
              Napi::Function::New(env, OfflineTtsSampleRateWrapper));

  exports.Set(Napi::String::New(env, "getOfflineTtsNumSpeakers"),
              Napi::Function::New(env, OfflineTtsNumSpeakersWrapper));

  exports.Set(Napi::String::New(env, "offlineTtsGenerate"),
              Napi::Function::New(env, OfflineTtsGenerateWrapper));
}
