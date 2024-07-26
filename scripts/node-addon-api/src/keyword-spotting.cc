// scripts/node-addon-api/src/keyword-spotting.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

// defined ./streaming-asr.cc
SherpaOnnxFeatureConfig GetFeatureConfig(Napi::Object obj);

// defined ./streaming-asr.cc
SherpaOnnxOnlineModelConfig GetOnlineModelConfig(Napi::Object obj);

static Napi::External<SherpaOnnxKeywordSpotter> CreateKeywordSpotterWrapper(
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
  SherpaOnnxKeywordSpotterConfig c;
  memset(&c, 0, sizeof(c));
  c.feat_config = GetFeatureConfig(o);
  c.model_config = GetOnlineModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_INT32(max_active_paths, maxActivePaths);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(num_trailing_blanks, numTrailingBlanks);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(keywords_score, keywordsScore);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(keywords_threshold, keywordsThreshold);
  SHERPA_ONNX_ASSIGN_ATTR_STR(keywords_file, keywordsFile);

  SherpaOnnxKeywordSpotter *kws = SherpaOnnxCreateKeywordSpotter(&c);

  if (c.model_config.transducer.encoder) {
    delete[] c.model_config.transducer.encoder;
  }

  if (c.model_config.transducer.decoder) {
    delete[] c.model_config.transducer.decoder;
  }

  if (c.model_config.transducer.joiner) {
    delete[] c.model_config.transducer.joiner;
  }

  if (c.model_config.paraformer.encoder) {
    delete[] c.model_config.paraformer.encoder;
  }

  if (c.model_config.paraformer.decoder) {
    delete[] c.model_config.paraformer.decoder;
  }

  if (c.model_config.zipformer2_ctc.model) {
    delete[] c.model_config.zipformer2_ctc.model;
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

  if (c.keywords_file) {
    delete[] c.keywords_file;
  }

  if (!kws) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxKeywordSpotter>::New(
      env, kws, [](Napi::Env env, SherpaOnnxKeywordSpotter *kws) {
        SherpaOnnxDestroyKeywordSpotter(kws);
      });
}

static Napi::External<SherpaOnnxOnlineStream> CreateKeywordStreamWrapper(
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
        env, "You should pass a keyword spotter pointer as the only argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxKeywordSpotter *kws =
      info[0].As<Napi::External<SherpaOnnxKeywordSpotter>>().Data();

  SherpaOnnxOnlineStream *stream = SherpaOnnxCreateKeywordStream(kws);

  return Napi::External<SherpaOnnxOnlineStream>::New(
      env, stream, [](Napi::Env env, SherpaOnnxOnlineStream *stream) {
        SherpaOnnxDestroyOnlineStream(stream);
      });
}

static Napi::Boolean IsKeywordStreamReadyWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a keyword spotter pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxKeywordSpotter *kws =
      info[0].As<Napi::External<SherpaOnnxKeywordSpotter>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  int32_t is_ready = SherpaOnnxIsKeywordStreamReady(kws, stream);

  return Napi::Boolean::New(env, is_ready);
}

static void DecodeKeywordStreamWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a keyword spotter pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxKeywordSpotter *kws =
      info[0].As<Napi::External<SherpaOnnxKeywordSpotter>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  SherpaOnnxDecodeKeywordStream(kws, stream);
}

static Napi::String GetKeywordResultAsJsonWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a keyword spotter pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxKeywordSpotter *kws =
      info[0].As<Napi::External<SherpaOnnxKeywordSpotter>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  const char *json = SherpaOnnxGetKeywordResultAsJson(kws, stream);

  Napi::String s = Napi::String::New(env, json);

  SherpaOnnxFreeKeywordResultJson(json);

  return s;
}

void InitKeywordSpotting(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createKeywordSpotter"),
              Napi::Function::New(env, CreateKeywordSpotterWrapper));

  exports.Set(Napi::String::New(env, "createKeywordStream"),
              Napi::Function::New(env, CreateKeywordStreamWrapper));

  exports.Set(Napi::String::New(env, "isKeywordStreamReady"),
              Napi::Function::New(env, IsKeywordStreamReadyWrapper));

  exports.Set(Napi::String::New(env, "decodeKeywordStream"),
              Napi::Function::New(env, DecodeKeywordStreamWrapper));

  exports.Set(Napi::String::New(env, "getKeywordResultAsJson"),
              Napi::Function::New(env, GetKeywordResultAsJsonWrapper));
}
