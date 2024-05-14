// scripts/node-addon-api/src/streaming-asr.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"
/*
{
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  }
};
 */
SherpaOnnxFeatureConfig GetFeatureConfig(Napi::Object obj) {
  SherpaOnnxFeatureConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("featConfig") || !obj.Get("featConfig").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("featConfig").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_INT32(sample_rate, sampleRate);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(feature_dim, featureDim);

  return c;
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
  SherpaOnnxOnlineTransducerModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("transducer") || !obj.Get("transducer").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("transducer").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(encoder, encoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(decoder, decoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(joiner, joiner);

  return c;
}

static SherpaOnnxOnlineZipformer2CtcModelConfig
GetOnlineZipformer2CtcModelConfig(Napi::Object obj) {
  SherpaOnnxOnlineZipformer2CtcModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("zipformer2Ctc") || !obj.Get("zipformer2Ctc").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("zipformer2Ctc").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);

  return c;
}

static SherpaOnnxOnlineParaformerModelConfig GetOnlineParaformerModelConfig(
    Napi::Object obj) {
  SherpaOnnxOnlineParaformerModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("paraformer") || !obj.Get("paraformer").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("paraformer").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(encoder, encoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(decoder, decoder);

  return c;
}

static SherpaOnnxOnlineModelConfig GetOnlineModelConfig(Napi::Object obj) {
  SherpaOnnxOnlineModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("modelConfig") || !obj.Get("modelConfig").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("modelConfig").As<Napi::Object>();

  c.transducer = GetOnlineTransducerModelConfig(o);
  c.paraformer = GetOnlineParaformerModelConfig(o);
  c.zipformer2_ctc = GetOnlineZipformer2CtcModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_STR(tokens, tokens);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(num_threads, numThreads);
  SHERPA_ONNX_ASSIGN_ATTR_STR(provider, provider);

  if (o.Has("debug") &&
      (o.Get("debug").IsNumber() || o.Get("debug").IsBoolean())) {
    if (o.Get("debug").IsBoolean()) {
      c.debug = o.Get("debug").As<Napi::Boolean>().Value();
    } else {
      c.debug = o.Get("debug").As<Napi::Number>().Int32Value();
    }
  }

  SHERPA_ONNX_ASSIGN_ATTR_STR(model_type, modelType);

  return c;
}

static SherpaOnnxOnlineCtcFstDecoderConfig GetCtcFstDecoderConfig(
    Napi::Object obj) {
  SherpaOnnxOnlineCtcFstDecoderConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("ctcFstDecoderConfig") ||
      !obj.Get("ctcFstDecoderConfig").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("ctcFstDecoderConfig").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(graph, graph);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(max_active, maxActive);

  return c;
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

  Napi::Object o = info[0].As<Napi::Object>();
  SherpaOnnxOnlineRecognizerConfig c;
  memset(&c, 0, sizeof(c));
  c.feat_config = GetFeatureConfig(o);
  c.model_config = GetOnlineModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_STR(decoding_method, decodingMethod);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(max_active_paths, maxActivePaths);

  // enableEndpoint can be either a boolean or an integer
  if (o.Has("enableEndpoint") && (o.Get("enableEndpoint").IsNumber() ||
                                  o.Get("enableEndpoint").IsBoolean())) {
    if (o.Get("enableEndpoint").IsNumber()) {
      c.enable_endpoint =
          o.Get("enableEndpoint").As<Napi::Number>().Int32Value();
    } else {
      c.enable_endpoint = o.Get("enableEndpoint").As<Napi::Boolean>().Value();
    }
  }

  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(rule1_min_trailing_silence,
                                rule1MinTrailingSilence);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(rule2_min_trailing_silence,
                                rule2MinTrailingSilence);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(rule3_min_utterance_length,
                                rule3MinUtteranceLength);
  SHERPA_ONNX_ASSIGN_ATTR_STR(hotwords_file, hotwordsFile);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(hotwords_score, hotwordsScore);

  c.ctc_fst_decoder_config = GetCtcFstDecoderConfig(o);

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

  if (c.decoding_method) {
    delete[] c.decoding_method;
  }

  if (c.hotwords_file) {
    delete[] c.hotwords_file;
  }

  if (c.ctc_fst_decoder_config.graph) {
    delete[] c.ctc_fst_decoder_config.graph;
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
        env,
        "You should pass an online recognizer pointer as the only argument")
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
    Napi::TypeError::New(env, "Argument 0 should be an online stream pointer.")
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
                         "Argument 0 should be an online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
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
                         "Argument 0 should be an online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
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
                         "Argument 0 should be an online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
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

static void InputFinishedWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxOnlineStream *stream =
      info[0].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  InputFinished(stream);
}

static void ResetOnlineStreamWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "Argument 0 should be an online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxOnlineRecognizer *recognizer =
      info[0].As<Napi::External<SherpaOnnxOnlineRecognizer>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  Reset(recognizer, stream);
}

static Napi::Boolean IsEndpointWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "Argument 0 should be an online recognizer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env, "Argument 1 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxOnlineRecognizer *recognizer =
      info[0].As<Napi::External<SherpaOnnxOnlineRecognizer>>().Data();

  SherpaOnnxOnlineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOnlineStream>>().Data();

  int32_t is_endpoint = IsEndpoint(recognizer, stream);

  return Napi::Boolean::New(env, is_endpoint);
}

static Napi::External<SherpaOnnxDisplay> CreateDisplayWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsNumber()) {
    Napi::TypeError::New(env, "Expect a number as the argument")
        .ThrowAsJavaScriptException();

    return {};
  }
  int32_t max_word_per_line = info[0].As<Napi::Number>().Int32Value();

  const SherpaOnnxDisplay *display = CreateDisplay(max_word_per_line);

  return Napi::External<SherpaOnnxDisplay>::New(
      env, const_cast<SherpaOnnxDisplay *>(display),
      [](Napi::Env env, SherpaOnnxDisplay *display) {
        DestroyDisplay(display);
      });
}

static void PrintWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 3) {
    std::ostringstream os;
    os << "Expect only 3 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an online stream pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!info[1].IsNumber()) {
    Napi::TypeError::New(env, "Argument 1 should be a number.")
        .ThrowAsJavaScriptException();

    return;
  }

  if (!info[2].IsString()) {
    Napi::TypeError::New(env, "Argument 2 should be a string.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxDisplay *display =
      info[0].As<Napi::External<SherpaOnnxDisplay>>().Data();

  int32_t idx = info[1].As<Napi::Number>().Int32Value();

  Napi::String text = info[2].As<Napi::String>();
  std::string s = text.Utf8Value();
  SherpaOnnxPrint(display, idx, s.c_str());
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

  exports.Set(Napi::String::New(env, "inputFinished"),
              Napi::Function::New(env, InputFinishedWrapper));

  exports.Set(Napi::String::New(env, "reset"),
              Napi::Function::New(env, ResetOnlineStreamWrapper));

  exports.Set(Napi::String::New(env, "isEndpoint"),
              Napi::Function::New(env, IsEndpointWrapper));

  exports.Set(Napi::String::New(env, "createDisplay"),
              Napi::Function::New(env, CreateDisplayWrapper));

  exports.Set(Napi::String::New(env, "print"),
              Napi::Function::New(env, PrintWrapper));
}
