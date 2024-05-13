// scripts/node-addon-api/src/non-streaming-asr.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <sstream>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

// defined in ./streaming-asr.cc
SherpaOnnxFeatureConfig GetFeatureConfig(Napi::Object obj);

static SherpaOnnxOfflineTransducerModelConfig GetOfflineTransducerModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTransducerModelConfig config;
  memset(&config, 0, sizeof(config));

  if (!obj.Has("transducer") || !obj.Get("transducer").IsObject()) {
    return config;
  }

  Napi::Object o = obj.Get("transducer").As<Napi::Object>();

  if (o.Has("encoder") && o.Get("encoder").IsString()) {
    Napi::String encoder = o.Get("encoder").As<Napi::String>();
    std::string s = encoder.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.encoder = p;
  }

  if (o.Has("decoder") && o.Get("decoder").IsString()) {
    Napi::String decoder = o.Get("decoder").As<Napi::String>();
    std::string s = decoder.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.decoder = p;
  }

  if (o.Has("joiner") && o.Get("joiner").IsString()) {
    Napi::String joiner = o.Get("joiner").As<Napi::String>();
    std::string s = joiner.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.joiner = p;
  }

  return config;
}

static SherpaOnnxOfflineParaformerModelConfig GetOfflineParaformerModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineParaformerModelConfig config;
  memset(&config, 0, sizeof(config));

  if (!obj.Has("paraformer") || !obj.Get("paraformer").IsObject()) {
    return config;
  }

  Napi::Object o = obj.Get("paraformer").As<Napi::Object>();

  if (o.Has("model") && o.Get("model").IsString()) {
    Napi::String model = o.Get("model").As<Napi::String>();
    std::string s = model.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.model = p;
  }

  return config;
}

static SherpaOnnxOfflineNemoEncDecCtcModelConfig GetOfflineNeMoCtcModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineNemoEncDecCtcModelConfig config;
  memset(&config, 0, sizeof(config));

  if (!obj.Has("nemoCtc") || !obj.Get("nemoCtc").IsObject()) {
    return config;
  }

  Napi::Object o = obj.Get("nemoCtc").As<Napi::Object>();

  if (o.Has("model") && o.Get("model").IsString()) {
    Napi::String model = o.Get("model").As<Napi::String>();
    std::string s = model.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.model = p;
  }

  return config;
}

static SherpaOnnxOfflineWhisperModelConfig GetOfflineWhisperModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineWhisperModelConfig config;
  memset(&config, 0, sizeof(config));

  if (!obj.Has("whisper") || !obj.Get("whisper").IsObject()) {
    return config;
  }

  Napi::Object o = obj.Get("whisper").As<Napi::Object>();

  if (o.Has("encoder") && o.Get("encoder").IsString()) {
    Napi::String encoder = o.Get("encoder").As<Napi::String>();
    std::string s = encoder.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.encoder = p;
  }

  if (o.Has("decoder") && o.Get("decoder").IsString()) {
    Napi::String decoder = o.Get("decoder").As<Napi::String>();
    std::string s = decoder.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.decoder = p;
  }

  if (o.Has("language") && o.Get("language").IsString()) {
    Napi::String language = o.Get("language").As<Napi::String>();
    std::string s = language.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.language = p;
  }

  if (o.Has("task") && o.Get("task").IsString()) {
    Napi::String task = o.Get("task").As<Napi::String>();
    std::string s = task.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.task = p;
  }

  return config;
}

static SherpaOnnxOfflineTdnnModelConfig GetOfflineTdnnModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTdnnModelConfig config;
  memset(&config, 0, sizeof(config));

  if (!obj.Has("tdnn") || !obj.Get("tdnn").IsObject()) {
    return config;
  }

  Napi::Object o = obj.Get("tdnn").As<Napi::Object>();

  if (o.Has("model") && o.Get("model").IsString()) {
    Napi::String model = o.Get("model").As<Napi::String>();
    std::string s = model.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.model = p;
  }

  return config;
}

static SherpaOnnxOfflineModelConfig GetOfflineModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("modelConfig") || !obj.Get("modelConfig").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("modelConfig").As<Napi::Object>();

  c.transducer = GetOfflineTransducerModelConfig(o);
  c.paraformer = GetOfflineParaformerModelConfig(o);
  c.nemo_ctc = GetOfflineNeMoCtcModelConfig(o);
  c.whisper = GetOfflineWhisperModelConfig(o);
  c.tdnn = GetOfflineTdnnModelConfig(o);

  if (o.Has("tokens") && o.Get("tokens").IsString()) {
    Napi::String tokens = o.Get("tokens").As<Napi::String>();
    std::string s = tokens.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.tokens = p;
  }

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

  if (o.Has("modelType") && o.Get("modelType").IsString()) {
    Napi::String model_type = o.Get("modelType").As<Napi::String>();
    std::string s = model_type.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.model_type = p;
  }

  return c;
}

static SherpaOnnxOfflineLMConfig GetOfflineLMConfig(Napi::Object obj) {
  SherpaOnnxOfflineLMConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("lmConfig") || !obj.Get("lmConfig").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("lmConfig").As<Napi::Object>();

  if (o.Has("model") && o.Get("model").IsString()) {
    Napi::String model = o.Get("model").As<Napi::String>();
    std::string s = model.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.model = p;
  }

  if (o.Has("scale") && o.Get("scale").IsNumber()) {
    c.scale = o.Get("scale").As<Napi::Number>().FloatValue();
  }

  return c;
}

static Napi::External<SherpaOnnxOfflineRecognizer>
CreateOfflineRecognizerWrapper(const Napi::CallbackInfo &info) {
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

  SherpaOnnxOfflineRecognizerConfig c;
  memset(&c, 0, sizeof(c));
  c.feat_config = GetFeatureConfig(o);
  c.model_config = GetOfflineModelConfig(o);
  c.lm_config = GetOfflineLMConfig(o);

  if (o.Has("decodingMethod") && o.Get("decodingMethod").IsString()) {
    Napi::String decoding_method = o.Get("decodingMethod").As<Napi::String>();
    std::string s = decoding_method.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.decoding_method = p;
  }

  if (o.Has("maxActivePaths") && o.Get("maxActivePaths").IsNumber()) {
    c.max_active_paths =
        o.Get("maxActivePaths").As<Napi::Number>().Int32Value();
  }

  if (o.Has("hotwordsFile") && o.Get("hotwordsFile").IsString()) {
    Napi::String hotwords_file = o.Get("hotwordsFile").As<Napi::String>();
    std::string s = hotwords_file.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    c.hotwords_file = p;
  }

  if (o.Has("hotwordsScore") && o.Get("hotwordsScore").IsNumber()) {
    c.hotwords_score = o.Get("hotwordsScore").As<Napi::Number>().FloatValue();
  }

  SherpaOnnxOfflineRecognizer *recognizer = CreateOfflineRecognizer(&c);

  if (c.model_config.transducer.encoder) {
    delete[] c.model_config.transducer.encoder;
  }

  if (c.model_config.transducer.decoder) {
    delete[] c.model_config.transducer.decoder;
  }

  if (c.model_config.transducer.joiner) {
    delete[] c.model_config.transducer.joiner;
  }

  if (c.model_config.paraformer.model) {
    delete[] c.model_config.paraformer.model;
  }

  if (c.model_config.nemo_ctc.model) {
    delete[] c.model_config.nemo_ctc.model;
  }

  if (c.model_config.whisper.encoder) {
    delete[] c.model_config.whisper.encoder;
  }

  if (c.model_config.whisper.decoder) {
    delete[] c.model_config.whisper.decoder;
  }

  if (c.model_config.whisper.language) {
    delete[] c.model_config.whisper.language;
  }

  if (c.model_config.whisper.task) {
    delete[] c.model_config.whisper.task;
  }

  if (c.model_config.tdnn.model) {
    delete[] c.model_config.tdnn.model;
  }

  if (c.model_config.tokens) {
    delete[] c.model_config.tokens;
  }

  if (c.model_config.provider) {
    delete[] c.model_config.provider;
  }

  if (c.model_config.model_type) {
    delete[] c.model_config.model_type;
  }

  if (c.lm_config.model) {
    delete[] c.lm_config.model;
  }

  if (c.decoding_method) {
    delete[] c.decoding_method;
  }

  if (c.hotwords_file) {
    delete[] c.hotwords_file;
  }

  if (!recognizer) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxOfflineRecognizer>::New(
      env, recognizer,
      [](Napi::Env env, SherpaOnnxOfflineRecognizer *recognizer) {
        DestroyOfflineRecognizer(recognizer);
      });
}

static Napi::External<SherpaOnnxOfflineStream> CreateOfflineStreamWrapper(
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
        "You should pass an offline recognizer pointer as the only argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxOfflineRecognizer *recognizer =
      info[0].As<Napi::External<SherpaOnnxOfflineRecognizer>>().Data();

  SherpaOnnxOfflineStream *stream = CreateOfflineStream(recognizer);

  return Napi::External<SherpaOnnxOfflineStream>::New(
      env, stream, [](Napi::Env env, SherpaOnnxOfflineStream *stream) {
        DestroyOfflineStream(stream);
      });
}

static void AcceptWaveformOfflineWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxOfflineStream *stream =
      info[0].As<Napi::External<SherpaOnnxOfflineStream>>().Data();

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Argument 1 should be an object")
        .ThrowAsJavaScriptException();

    return;
  }

  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("samples")) {
    Napi::TypeError::New(env, "The argument object should have a field samples")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!obj.Get("samples").IsTypedArray()) {
    Napi::TypeError::New(env, "The object['samples'] should be a typed array")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!obj.Has("sampleRate")) {
    Napi::TypeError::New(env,
                         "The argument object should have a field sampleRate")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!obj.Get("sampleRate").IsNumber()) {
    Napi::TypeError::New(env, "The object['samples'] should be a number")
        .ThrowAsJavaScriptException();

    return;
  }

  Napi::Float32Array samples = obj.Get("samples").As<Napi::Float32Array>();
  int32_t sample_rate = obj.Get("sampleRate").As<Napi::Number>().Int32Value();

  AcceptWaveformOffline(stream, sample_rate, samples.Data(),
                        samples.ElementLength());
}

static void DecodeOfflineStreamWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "Argument 0 should be an offline recognizer pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an offline stream pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxOfflineRecognizer *recognizer =
      info[0].As<Napi::External<SherpaOnnxOfflineRecognizer>>().Data();

  SherpaOnnxOfflineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOfflineStream>>().Data();

  DecodeOfflineStream(recognizer, stream);
}

static Napi::String GetOfflineStreamResultAsJsonWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxOfflineStream *stream =
      info[0].As<Napi::External<SherpaOnnxOfflineStream>>().Data();

  const char *json = GetOfflineStreamResultAsJson(stream);
  Napi::String s = Napi::String::New(env, json);

  DestroyOfflineStreamResultJson(json);

  return s;
}

void InitNonStreamingAsr(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOfflineRecognizer"),
              Napi::Function::New(env, CreateOfflineRecognizerWrapper));

  exports.Set(Napi::String::New(env, "createOfflineStream"),
              Napi::Function::New(env, CreateOfflineStreamWrapper));

  exports.Set(Napi::String::New(env, "acceptWaveformOffline"),
              Napi::Function::New(env, AcceptWaveformOfflineWrapper));

  exports.Set(Napi::String::New(env, "decodeOfflineStream"),
              Napi::Function::New(env, DecodeOfflineStreamWrapper));

  exports.Set(Napi::String::New(env, "getOfflineStreamResultAsJson"),
              Napi::Function::New(env, GetOfflineStreamResultAsJsonWrapper));
}
