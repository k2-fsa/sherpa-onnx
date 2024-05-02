// scripts/node-addon-api/src/streaming-asr.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <sstream>

#include "napi.h"
#include "sherpa-onnx/c-api/c-api.h"
/*
const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  }
};
 */
static SherpaOnnxFeatureConfig GetFeatureConfig(Napi::Object obj) {
  SherpaOnnxFeatureConfig config;
  config.sample_rate = 16000;
  config.feature_dim = 80;

  if (obj.Has("featConfig") && obj.Get("featConfig").IsObject()) {
    Napi::Object featConfig = obj.Get("featConfig").As<Napi::Object>();

    if (featConfig.Has("sampleRate") &&
        featConfig.Get("sampleRate").IsNumber()) {
      config.sample_rate =
          featConfig.Get("sampleRate").As<Napi::Number>().Int32Value();
    }

    if (featConfig.Has("featureDim") &&
        featConfig.Get("featureDim").IsNumber()) {
      config.feature_dim =
          featConfig.Get("featureDim").As<Napi::Number>().Int32Value();
    }
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
  SherpaOnnxOnlineTransducerModelConfig transducer;
  memset(&transducer, 0, sizeof(transducer));

  if (obj.Has("transducer") && obj.Get("transducer").IsObject()) {
    Napi::Object o = obj.Get("transducer").As<Napi::Object>();

    if (o.Has("encoder") && o.Get("encoder").IsString()) {
      Napi::String encoder = o.Get("encoder").As<Napi::String>();
      std::string s = encoder.Utf8Value();
      char *p = new char[s.size() + 1];
      std::copy(s.begin(), s.end(), p);
      p[s.size()] = 0;

      transducer.encoder = p;
    }

    if (o.Has("decoder") && o.Get("decoder").IsString()) {
      Napi::String decoder = o.Get("decoder").As<Napi::String>();
      std::string s = decoder.Utf8Value();
      char *p = new char[s.size() + 1];
      std::copy(s.begin(), s.end(), p);
      p[s.size()] = 0;

      transducer.decoder = p;
    }

    if (o.Has("joiner") && o.Get("joiner").IsString()) {
      Napi::String joiner = o.Get("joiner").As<Napi::String>();
      std::string s = joiner.Utf8Value();
      char *p = new char[s.size() + 1];
      std::copy(s.begin(), s.end(), p);
      p[s.size()] = 0;

      transducer.joiner = p;
    }
  }

  return transducer;
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

    Napi::TypeError::New(env, os.str().c_str()).ThrowAsJavaScriptException();

    return Napi::External<SherpaOnnxOnlineRecognizer>::New(env, nullptr);
  }

  if (!info[0].IsObject()) {
    Napi::TypeError::New(env, "Expect an object as the argument")
        .ThrowAsJavaScriptException();

    return Napi::External<SherpaOnnxOnlineRecognizer>::New(env, nullptr);
  }

  Napi::Object config = info[0].As<Napi::Object>();
  SherpaOnnxOnlineRecognizerConfig c;
  c.feat_config = GetFeatureConfig(config);
  c.model_config = GetOnlineModelConfig(config);
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

  return Napi::External<SherpaOnnxOnlineRecognizer>::New(env, nullptr);
}

void InitStreamingAsr(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOnlineRecognizer"),
              Napi::Function::New(env, CreateOnlineRecognizerWrapper));
}
