// scripts/node-addon-api/src/streaming-asr.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <sstream>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"
/*
{
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  }
};
 */
static SherpaOnnxFeatureConfig GetFeatureConfig(Napi::Object obj) {
  SherpaOnnxFeatureConfig config;
  memset(&config, 0, sizeof(config));

  if (!obj.Has("featConfig") || !obj.Get("featConfig").IsObject()) {
    return config;
  }

  Napi::Object featConfig = obj.Get("featConfig").As<Napi::Object>();

  if (featConfig.Has("sampleRate") && featConfig.Get("sampleRate").IsNumber()) {
    config.sample_rate =
        featConfig.Get("sampleRate").As<Napi::Number>().Int32Value();
  }

  if (featConfig.Has("featureDim") && featConfig.Get("featureDim").IsNumber()) {
    config.feature_dim =
        featConfig.Get("featureDim").As<Napi::Number>().Int32Value();
  }

  return config;
}
/*
{
  'transducer': {
    'encoder': './encoder.onnx',
    'decoder': './decoder.onnx',
    'joiner': './joiner.onnx',
  }
}
 */

static SherpaOnnxOnlineTransducerModelConfig GetOnlineTransducerModelConfig(
    Napi::Object obj) {
  SherpaOnnxOnlineTransducerModelConfig config;
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

static SherpaOnnxOnlineModelConfig GetOnlineModelConfig(Napi::Object obj) {
  SherpaOnnxOnlineModelConfig config;
  memset(&config, 0, sizeof(config));

  if (!obj.Has("modelConfig") || !obj.Get("modelConfig").IsObject()) {
    return config;
  }

  Napi::Object o = obj.Get("modelConfig").As<Napi::Object>();

  config.transducer = GetOnlineTransducerModelConfig(o);

  if (o.Has("tokens") && o.Get("tokens").IsString()) {
    Napi::String tokens = o.Get("tokens").As<Napi::String>();
    std::string s = tokens.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.tokens = p;
  }

  if (o.Has("numThreads") && o.Get("numThreads").IsNumber()) {
    config.num_threads = o.Get("numThreads").As<Napi::Number>().Int32Value();
  }

  if (o.Has("provider") && o.Get("provider").IsString()) {
    Napi::String provider = o.Get("provider").As<Napi::String>();
    std::string s = provider.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.provider = p;
  }

  if (o.Has("debug") && o.Get("debug").IsNumber()) {
    config.debug = o.Get("debug").As<Napi::Number>().Int32Value();
  }

  if (o.Has("modelType") && o.Get("modelType").IsString()) {
    Napi::String model_type = o.Get("modelType").As<Napi::String>();
    std::string s = model_type.Utf8Value();
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = 0;

    config.model_type = p;
  }

  return config;
}

static Napi::External<SherpaOnnxOnlineRecognizer> CreateOnlineRecognizerWrapper(
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

  Napi::Object config = info[0].As<Napi::Object>();
  SherpaOnnxOnlineRecognizerConfig c;
  memset(&c, 0, sizeof(c));
  c.feat_config = GetFeatureConfig(config);
  c.model_config = GetOnlineModelConfig(config);
#if 0
  printf("encoder: %s\n", c.model_config.transducer.encoder
                              ? c.model_config.transducer.encoder
                              : "no");
  printf("decoder: %s\n", c.model_config.transducer.decoder
                              ? c.model_config.transducer.decoder
                              : "no");
  printf("joiner: %s\n", c.model_config.transducer.joiner
                             ? c.model_config.transducer.joiner
                             : "no");

  printf("tokens: %s\n", c.model_config.tokens ? c.model_config.tokens : "no");
  printf("num_threads: %d\n", c.model_config.num_threads);
  printf("provider: %s\n",
         c.model_config.provider ? c.model_config.provider : "no");
  printf("debug: %d\n", c.model_config.debug);
  printf("model_type: %s\n",
         c.model_config.model_type ? c.model_config.model_type : "no");
#endif

  SherpaOnnxOnlineRecognizer *recognizer = CreateOnlineRecognizer(&c);

  if (c.model_config.transducer.encoder) {
    delete[] c.model_config.transducer.encoder;
  }

  if (c.model_config.transducer.decoder) {
    delete[] c.model_config.transducer.decoder;
  }

  if (c.model_config.transducer.joiner) {
    delete[] c.model_config.transducer.joiner;
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

  if (!recognizer) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxOnlineRecognizer>::New(
      env, recognizer,
      [](Napi::Env env, SherpaOnnxOnlineRecognizer *recognizer) {
        DestroyOnlineRecognizer(recognizer);
      });
}

static Napi::External<SherpaOnnxOnlineStream> CreateOnlineStreamWrapper(
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
        env, "You should pass a recognizer pointer as the only argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxOnlineRecognizer *recognizer =
      info[0].As<Napi::External<SherpaOnnxOnlineRecognizer>>().Data();

  SherpaOnnxOnlineStream *stream = CreateOnlineStream(recognizer);

  return Napi::External<SherpaOnnxOnlineStream>::New(
      env, stream, [](Napi::Env env, SherpaOnnxOnlineStream *stream) {
        DestroyOnlineStream(stream);
      });
}

static void AcceptWaveformWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a online stream pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxOnlineStream *stream =
      info[0].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

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

  AcceptWaveform(stream, sample_rate, samples.Data(), samples.ElementLength());
}

static Napi::Boolean IsOnlineStreamReadyWrapper(
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
                         "Argument 0 should be a online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env,
                         "Argument 1 should be a online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxOnlineRecognizer *recognizer =
      info[0].As<Napi::External<SherpaOnnxOnlineRecognizer>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  int32_t is_ready = IsOnlineStreamReady(recognizer, stream);

  return Napi::Boolean::New(env, is_ready);
}

static void DecodeOnlineStreamWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "Argument 0 should be a online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env,
                         "Argument 1 should be a online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxOnlineRecognizer *recognizer =
      info[0].As<Napi::External<SherpaOnnxOnlineRecognizer>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  DecodeOnlineStream(recognizer, stream);
}

static Napi::String GetOnlineStreamResultAsJsonWrapper(
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
                         "Argument 0 should be a online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env,
                         "Argument 1 should be a online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxOnlineRecognizer *recognizer =
      info[0].As<Napi::External<SherpaOnnxOnlineRecognizer>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  const char *json = GetOnlineStreamResultAsJson(recognizer, stream);
  Napi::String s = Napi::String::New(env, json);

  DestroyOnlineStreamResultJson(json);

  return s;
}

void InitStreamingAsr(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOnlineRecognizer"),
              Napi::Function::New(env, CreateOnlineRecognizerWrapper));

  exports.Set(Napi::String::New(env, "createOnlineStream"),
              Napi::Function::New(env, CreateOnlineStreamWrapper));

  exports.Set(Napi::String::New(env, "acceptWaveformOnline"),
              Napi::Function::New(env, AcceptWaveformWrapper));

  exports.Set(Napi::String::New(env, "isOnlineStreamReady"),
              Napi::Function::New(env, IsOnlineStreamReadyWrapper));

  exports.Set(Napi::String::New(env, "decodeOnlineStream"),
              Napi::Function::New(env, DecodeOnlineStreamWrapper));

  exports.Set(Napi::String::New(env, "getOnlineStreamResultAsJson"),
              Napi::Function::New(env, GetOnlineStreamResultAsJsonWrapper));
}
