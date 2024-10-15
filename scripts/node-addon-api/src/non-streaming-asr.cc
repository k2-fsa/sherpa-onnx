// scripts/node-addon-api/src/non-streaming-asr.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

// defined in ./streaming-asr.cc
SherpaOnnxFeatureConfig GetFeatureConfig(Napi::Object obj);

static SherpaOnnxOfflineTransducerModelConfig GetOfflineTransducerModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTransducerModelConfig c;
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

static SherpaOnnxOfflineParaformerModelConfig GetOfflineParaformerModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineParaformerModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("paraformer") || !obj.Get("paraformer").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("paraformer").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);

  return c;
}

static SherpaOnnxOfflineNemoEncDecCtcModelConfig GetOfflineNeMoCtcModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineNemoEncDecCtcModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("nemoCtc") || !obj.Get("nemoCtc").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("nemoCtc").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);

  return c;
}

static SherpaOnnxOfflineWhisperModelConfig GetOfflineWhisperModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineWhisperModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("whisper") || !obj.Get("whisper").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("whisper").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(encoder, encoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(decoder, decoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(language, language);
  SHERPA_ONNX_ASSIGN_ATTR_STR(task, task);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(tail_paddings, tailPaddings);

  return c;
}

static SherpaOnnxOfflineTdnnModelConfig GetOfflineTdnnModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTdnnModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("tdnn") || !obj.Get("tdnn").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("tdnn").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);

  return c;
}

static SherpaOnnxOfflineSenseVoiceModelConfig GetOfflineSenseVoiceModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineSenseVoiceModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("senseVoice") || !obj.Get("senseVoice").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("senseVoice").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  SHERPA_ONNX_ASSIGN_ATTR_STR(language, language);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(use_itn, useInverseTextNormalization);

  return c;
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
  c.sense_voice = GetOfflineSenseVoiceModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_STR(tokens, tokens);
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
  SHERPA_ONNX_ASSIGN_ATTR_STR(model_type, modelType);
  SHERPA_ONNX_ASSIGN_ATTR_STR(modeling_unit, modelingUnit);
  SHERPA_ONNX_ASSIGN_ATTR_STR(bpe_vocab, bpeVocab);
  SHERPA_ONNX_ASSIGN_ATTR_STR(telespeech_ctc, teleSpeechCtc);

  return c;
}

static SherpaOnnxOfflineLMConfig GetOfflineLMConfig(Napi::Object obj) {
  SherpaOnnxOfflineLMConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("lmConfig") || !obj.Get("lmConfig").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("lmConfig").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(scale, scale);

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

  SHERPA_ONNX_ASSIGN_ATTR_STR(decoding_method, decodingMethod);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(max_active_paths, maxActivePaths);
  SHERPA_ONNX_ASSIGN_ATTR_STR(hotwords_file, hotwordsFile);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(hotwords_score, hotwordsScore);
  SHERPA_ONNX_ASSIGN_ATTR_STR(rule_fsts, ruleFsts);
  SHERPA_ONNX_ASSIGN_ATTR_STR(rule_fars, ruleFars);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(blank_penalty, blankPenalty);

  SherpaOnnxOfflineRecognizer *recognizer =
      SherpaOnnxCreateOfflineRecognizer(&c);

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

  if (c.model_config.sense_voice.model) {
    delete[] c.model_config.sense_voice.model;
  }

  if (c.model_config.sense_voice.language) {
    delete[] c.model_config.sense_voice.language;
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

  if (c.model_config.modeling_unit) {
    delete[] c.model_config.modeling_unit;
  }

  if (c.model_config.bpe_vocab) {
    delete[] c.model_config.bpe_vocab;
  }

  if (c.model_config.telespeech_ctc) {
    delete[] c.model_config.telespeech_ctc;
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

  if (c.rule_fsts) {
    delete[] c.rule_fsts;
  }

  if (c.rule_fars) {
    delete[] c.rule_fars;
  }

  if (!recognizer) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxOfflineRecognizer>::New(
      env, recognizer,
      [](Napi::Env env, SherpaOnnxOfflineRecognizer *recognizer) {
        SherpaOnnxDestroyOfflineRecognizer(recognizer);
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

  SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(recognizer);

  return Napi::External<SherpaOnnxOfflineStream>::New(
      env, stream, [](Napi::Env env, SherpaOnnxOfflineStream *stream) {
        SherpaOnnxDestroyOfflineStream(stream);
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

  SherpaOnnxAcceptWaveformOffline(stream, sample_rate, samples.Data(),
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

  SherpaOnnxDecodeOfflineStream(recognizer, stream);
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

  const char *json = SherpaOnnxGetOfflineStreamResultAsJson(stream);
  Napi::String s = Napi::String::New(env, json);

  SherpaOnnxDestroyOfflineStreamResultJson(json);

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
