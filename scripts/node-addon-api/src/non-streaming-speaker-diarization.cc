// scripts/node-addon-api/src/non-streaming-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <algorithm>
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig
GetOfflineSpeakerSegmentationPyannoteModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("pyannote") || !obj.Get("pyannote").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("pyannote").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);

  return c;
}

static SherpaOnnxOfflineSpeakerSegmentationModelConfig
GetOfflineSpeakerSegmentationModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineSpeakerSegmentationModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("segmentation") || !obj.Get("segmentation").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("segmentation").As<Napi::Object>();

  c.pyannote = GetOfflineSpeakerSegmentationPyannoteModelConfig(o);

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

static SherpaOnnxSpeakerEmbeddingExtractorConfig
GetSpeakerEmbeddingExtractorConfig(Napi::Object obj) {
  SherpaOnnxSpeakerEmbeddingExtractorConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("embedding") || !obj.Get("embedding").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("embedding").As<Napi::Object>();

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

  return c;
}

static SherpaOnnxFastClusteringConfig GetFastClusteringConfig(
    Napi::Object obj) {
  SherpaOnnxFastClusteringConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("clustering") || !obj.Get("clustering").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("clustering").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_INT32(num_clusters, numClusters);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(threshold, threshold);

  return c;
}

static Napi::External<SherpaOnnxOfflineSpeakerDiarization>
CreateOfflineSpeakerDiarizationWrapper(const Napi::CallbackInfo &info) {
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

  SherpaOnnxOfflineSpeakerDiarizationConfig c;
  memset(&c, 0, sizeof(c));

  c.segmentation = GetOfflineSpeakerSegmentationModelConfig(o);
  c.embedding = GetSpeakerEmbeddingExtractorConfig(o);
  c.clustering = GetFastClusteringConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(min_duration_on, minDurationOn);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(min_duration_off, minDurationOff);

  const SherpaOnnxOfflineSpeakerDiarization *sd =
      SherpaOnnxCreateOfflineSpeakerDiarization(&c);

  if (c.segmentation.pyannote.model) {
    delete[] c.segmentation.pyannote.model;
  }

  if (c.segmentation.provider) {
    delete[] c.segmentation.provider;
  }

  if (c.embedding.model) {
    delete[] c.embedding.model;
  }

  if (c.embedding.provider) {
    delete[] c.embedding.provider;
  }

  if (!sd) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxOfflineSpeakerDiarization>::New(
      env, const_cast<SherpaOnnxOfflineSpeakerDiarization *>(sd),
      [](Napi::Env env, SherpaOnnxOfflineSpeakerDiarization *sd) {
        SherpaOnnxDestroyOfflineSpeakerDiarization(sd);
      });
}

static Napi::Number OfflineSpeakerDiarizationGetSampleRateWrapper(
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
        env, "Argument 0 should be an offline speaker diarization pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOfflineSpeakerDiarization *sd =
      info[0].As<Napi::External<SherpaOnnxOfflineSpeakerDiarization>>().Data();

  int32_t sample_rate = SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(sd);

  return Napi::Number::New(env, sample_rate);
}

static Napi::Array OfflineSpeakerDiarizationProcessWrapper(
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
        env, "Argument 0 should be an offline speaker diarization pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOfflineSpeakerDiarization *sd =
      info[0].As<Napi::External<SherpaOnnxOfflineSpeakerDiarization>>().Data();

  if (!info[1].IsTypedArray()) {
    Napi::TypeError::New(env, "Argument 1 should be a typed array")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Float32Array samples = info[1].As<Napi::Float32Array>();

  const SherpaOnnxOfflineSpeakerDiarizationResult *r =
      SherpaOnnxOfflineSpeakerDiarizationProcess(sd, samples.Data(),
                                                 samples.ElementLength());

  int32_t num_segments =
      SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(r);

  const SherpaOnnxOfflineSpeakerDiarizationSegment *segments =
      SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(r);

  Napi::Array ans = Napi::Array::New(env, num_segments);

  for (int32_t i = 0; i != num_segments; ++i) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set(Napi::String::New(env, "start"), segments[i].start);
    obj.Set(Napi::String::New(env, "end"), segments[i].end);
    obj.Set(Napi::String::New(env, "speaker"), segments[i].speaker);

    ans[i] = obj;
  }

  SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments);
  SherpaOnnxOfflineSpeakerDiarizationDestroyResult(r);

  return ans;
}

static void OfflineSpeakerDiarizationSetConfigWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(
        env, "Argument 0 should be an offline speaker diarization pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  const SherpaOnnxOfflineSpeakerDiarization *sd =
      info[0].As<Napi::External<SherpaOnnxOfflineSpeakerDiarization>>().Data();

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Expect an object as the argument")
        .ThrowAsJavaScriptException();

    return;
  }

  Napi::Object o = info[0].As<Napi::Object>();

  SherpaOnnxOfflineSpeakerDiarizationConfig c;
  memset(&c, 0, sizeof(c));

  c.clustering = GetFastClusteringConfig(o);
  SherpaOnnxOfflineSpeakerDiarizationSetConfig(sd, &c);
}

void InitNonStreamingSpeakerDiarization(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOfflineSpeakerDiarization"),
              Napi::Function::New(env, CreateOfflineSpeakerDiarizationWrapper));

  exports.Set(
      Napi::String::New(env, "getOfflineSpeakerDiarizationSampleRate"),
      Napi::Function::New(env, OfflineSpeakerDiarizationGetSampleRateWrapper));

  exports.Set(
      Napi::String::New(env, "offlineSpeakerDiarizationProcess"),
      Napi::Function::New(env, OfflineSpeakerDiarizationProcessWrapper));

  exports.Set(
      Napi::String::New(env, "offlineSpeakerDiarizationSetConfig"),
      Napi::Function::New(env, OfflineSpeakerDiarizationSetConfigWrapper));
}
