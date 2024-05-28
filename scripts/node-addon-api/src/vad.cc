// scripts/node-addon-api/src/vad.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <algorithm>
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static Napi::External<SherpaOnnxCircularBuffer> CreateCircularBufferWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsNumber()) {
    Napi::TypeError::New(env, "You should pass an integer as the argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxCircularBuffer *buf =
      SherpaOnnxCreateCircularBuffer(info[0].As<Napi::Number>().Int32Value());

  return Napi::External<SherpaOnnxCircularBuffer>::New(
      env, buf, [](Napi::Env env, SherpaOnnxCircularBuffer *p) {
        SherpaOnnxDestroyCircularBuffer(p);
      });
}

static void CircularBufferPushWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an CircularBuffer pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxCircularBuffer *buf =
      info[0].As<Napi::External<SherpaOnnxCircularBuffer>>().Data();

  if (!info[1].IsTypedArray()) {
    Napi::TypeError::New(env, "Argument 1 should be a Float32Array.")
        .ThrowAsJavaScriptException();

    return;
  }

  Napi::Float32Array data = info[1].As<Napi::Float32Array>();
  SherpaOnnxCircularBufferPush(buf, data.Data(), data.ElementLength());
}

// see https://github.com/nodejs/node-addon-api/blob/main/doc/typed_array.md
// https://github.com/nodejs/node-addon-examples/blob/main/src/2-js-to-native-conversion/typed_array_to_native/node-addon-api/typed_array_to_native.cc
static Napi::Float32Array CircularBufferGetWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 3 && info.Length() != 4) {
    std::ostringstream os;
    os << "Expect only 3 or 4 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an CircularBuffer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxCircularBuffer *buf =
      info[0].As<Napi::External<SherpaOnnxCircularBuffer>>().Data();

  if (!info[1].IsNumber()) {
    Napi::TypeError::New(env, "Argument 1 should be an integer (startIndex).")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[2].IsNumber()) {
    Napi::TypeError::New(env, "Argument 2 should be an integer (n).")
        .ThrowAsJavaScriptException();

    return {};
  }

  bool enable_external_buffer = true;
  if (info.Length() == 4) {
    if (info[3].IsBoolean()) {
      enable_external_buffer = info[3].As<Napi::Boolean>().Value();
    } else {
      Napi::TypeError::New(env, "Argument 3 should be a boolean.")
          .ThrowAsJavaScriptException();
    }
  }

  int32_t start_index = info[1].As<Napi::Number>().Int32Value();
  int32_t n = info[2].As<Napi::Number>().Int32Value();

  const float *data = SherpaOnnxCircularBufferGet(buf, start_index, n);

  if (enable_external_buffer) {
    Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
        env, const_cast<float *>(data), sizeof(float) * n,
        [](Napi::Env /*env*/, void *p) {
          SherpaOnnxCircularBufferFree(reinterpret_cast<const float *>(p));
        });

    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, n, arrayBuffer, 0);

    return float32Array;
  } else {
    // don't use external buffer
    Napi::ArrayBuffer arrayBuffer =
        Napi::ArrayBuffer::New(env, sizeof(float) * n);

    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, n, arrayBuffer, 0);

    std::copy(data, data + n, float32Array.Data());

    SherpaOnnxCircularBufferFree(data);

    return float32Array;
  }
}

static void CircularBufferPopWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an CircularBuffer pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxCircularBuffer *buf =
      info[0].As<Napi::External<SherpaOnnxCircularBuffer>>().Data();

  if (!info[1].IsNumber()) {
    Napi::TypeError::New(env, "Argument 1 should be an integer (n).")
        .ThrowAsJavaScriptException();

    return;
  }

  int32_t n = info[1].As<Napi::Number>().Int32Value();

  SherpaOnnxCircularBufferPop(buf, n);
}

static Napi::Number CircularBufferSizeWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an CircularBuffer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxCircularBuffer *buf =
      info[0].As<Napi::External<SherpaOnnxCircularBuffer>>().Data();

  int32_t size = SherpaOnnxCircularBufferSize(buf);

  return Napi::Number::New(env, size);
}

static Napi::Number CircularBufferHeadWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an CircularBuffer pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxCircularBuffer *buf =
      info[0].As<Napi::External<SherpaOnnxCircularBuffer>>().Data();

  int32_t size = SherpaOnnxCircularBufferHead(buf);

  return Napi::Number::New(env, size);
}

static void CircularBufferResetWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an CircularBuffer pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxCircularBuffer *buf =
      info[0].As<Napi::External<SherpaOnnxCircularBuffer>>().Data();

  SherpaOnnxCircularBufferReset(buf);
}

static SherpaOnnxSileroVadModelConfig GetSileroVadConfig(
    const Napi::Object &obj) {
  SherpaOnnxSileroVadModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("sileroVad") || !obj.Get("sileroVad").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("sileroVad").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(threshold, threshold);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(min_silence_duration, minSilenceDuration);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(min_speech_duration, minSpeechDuration);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(window_size, windowSize);

  return c;
}

static Napi::External<SherpaOnnxVoiceActivityDetector>
CreateVoiceActivityDetectorWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsObject()) {
    Napi::TypeError::New(env,
                         "You should pass an object as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!info[1].IsNumber()) {
    Napi::TypeError::New(env,
                         "You should pass an integer as the second argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Object o = info[0].As<Napi::Object>();

  SherpaOnnxVadModelConfig c;
  memset(&c, 0, sizeof(c));
  c.silero_vad = GetSileroVadConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_INT32(sample_rate, sampleRate);
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

  float buffer_size_in_seconds = info[1].As<Napi::Number>().FloatValue();

  SherpaOnnxVoiceActivityDetector *vad =
      SherpaOnnxCreateVoiceActivityDetector(&c, buffer_size_in_seconds);

  if (c.silero_vad.model) {
    delete[] c.silero_vad.model;
  }

  if (c.provider) {
    delete[] c.provider;
  }

  return Napi::External<SherpaOnnxVoiceActivityDetector>::New(
      env, vad, [](Napi::Env env, SherpaOnnxVoiceActivityDetector *p) {
        SherpaOnnxDestroyVoiceActivityDetector(p);
      });
}

static void VoiceActivityDetectorAcceptWaveformWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a VAD pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxVoiceActivityDetector *vad =
      info[0].As<Napi::External<SherpaOnnxVoiceActivityDetector>>().Data();

  if (!info[1].IsTypedArray()) {
    Napi::TypeError::New(
        env, "Argument 1 should be a Float32Array containing samples")
        .ThrowAsJavaScriptException();

    return;
  }

  Napi::Float32Array samples = info[1].As<Napi::Float32Array>();

  SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, samples.Data(),
                                                samples.ElementLength());
}

static Napi::Boolean VoiceActivityDetectorEmptyWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a VAD pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxVoiceActivityDetector *vad =
      info[0].As<Napi::External<SherpaOnnxVoiceActivityDetector>>().Data();

  int32_t is_empty = SherpaOnnxVoiceActivityDetectorEmpty(vad);

  return Napi::Boolean::New(env, is_empty);
}

static Napi::Boolean VoiceActivityDetectorDetectedWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a VAD pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxVoiceActivityDetector *vad =
      info[0].As<Napi::External<SherpaOnnxVoiceActivityDetector>>().Data();

  int32_t is_detected = SherpaOnnxVoiceActivityDetectorDetected(vad);

  return Napi::Boolean::New(env, is_detected);
}

static void VoiceActivityDetectorPopWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a VAD pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxVoiceActivityDetector *vad =
      info[0].As<Napi::External<SherpaOnnxVoiceActivityDetector>>().Data();

  SherpaOnnxVoiceActivityDetectorPop(vad);
}

static void VoiceActivityDetectorClearWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a VAD pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxVoiceActivityDetector *vad =
      info[0].As<Napi::External<SherpaOnnxVoiceActivityDetector>>().Data();

  SherpaOnnxVoiceActivityDetectorClear(vad);
}

static Napi::Object VoiceActivityDetectorFrontWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1 && info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 1 or 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a VAD pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  bool enable_external_buffer = true;
  if (info.Length() == 2) {
    if (info[1].IsBoolean()) {
      enable_external_buffer = info[1].As<Napi::Boolean>().Value();
    } else {
      Napi::TypeError::New(env, "Argument 1 should be a boolean.")
          .ThrowAsJavaScriptException();
    }
  }

  SherpaOnnxVoiceActivityDetector *vad =
      info[0].As<Napi::External<SherpaOnnxVoiceActivityDetector>>().Data();

  const SherpaOnnxSpeechSegment *segment =
      SherpaOnnxVoiceActivityDetectorFront(vad);

  if (enable_external_buffer) {
    Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
        env, const_cast<float *>(segment->samples), sizeof(float) * segment->n,
        [](Napi::Env /*env*/, void * /*data*/,
           const SherpaOnnxSpeechSegment *hint) {
          SherpaOnnxDestroySpeechSegment(hint);
        },
        segment);

    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, segment->n, arrayBuffer, 0);

    Napi::Object obj = Napi::Object::New(env);
    obj.Set(Napi::String::New(env, "start"), segment->start);
    obj.Set(Napi::String::New(env, "samples"), float32Array);

    return obj;
  } else {
    Napi::ArrayBuffer arrayBuffer =
        Napi::ArrayBuffer::New(env, sizeof(float) * segment->n);

    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, segment->n, arrayBuffer, 0);

    std::copy(segment->samples, segment->samples + segment->n,
              float32Array.Data());

    Napi::Object obj = Napi::Object::New(env);
    obj.Set(Napi::String::New(env, "start"), segment->start);
    obj.Set(Napi::String::New(env, "samples"), float32Array);

    SherpaOnnxDestroySpeechSegment(segment);

    return obj;
  }
}

static void VoiceActivityDetectorResetWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return;
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be a VAD pointer.")
        .ThrowAsJavaScriptException();

    return;
  }

  SherpaOnnxVoiceActivityDetector *vad =
      info[0].As<Napi::External<SherpaOnnxVoiceActivityDetector>>().Data();

  SherpaOnnxVoiceActivityDetectorReset(vad);
}

void InitVad(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createCircularBuffer"),
              Napi::Function::New(env, CreateCircularBufferWrapper));

  exports.Set(Napi::String::New(env, "circularBufferPush"),
              Napi::Function::New(env, CircularBufferPushWrapper));

  exports.Set(Napi::String::New(env, "circularBufferGet"),
              Napi::Function::New(env, CircularBufferGetWrapper));

  exports.Set(Napi::String::New(env, "circularBufferPop"),
              Napi::Function::New(env, CircularBufferPopWrapper));

  exports.Set(Napi::String::New(env, "circularBufferSize"),
              Napi::Function::New(env, CircularBufferSizeWrapper));

  exports.Set(Napi::String::New(env, "circularBufferHead"),
              Napi::Function::New(env, CircularBufferHeadWrapper));

  exports.Set(Napi::String::New(env, "circularBufferReset"),
              Napi::Function::New(env, CircularBufferResetWrapper));

  exports.Set(Napi::String::New(env, "createVoiceActivityDetector"),
              Napi::Function::New(env, CreateVoiceActivityDetectorWrapper));

  exports.Set(
      Napi::String::New(env, "voiceActivityDetectorAcceptWaveform"),
      Napi::Function::New(env, VoiceActivityDetectorAcceptWaveformWrapper));

  exports.Set(Napi::String::New(env, "voiceActivityDetectorIsEmpty"),
              Napi::Function::New(env, VoiceActivityDetectorEmptyWrapper));

  exports.Set(Napi::String::New(env, "voiceActivityDetectorIsDetected"),
              Napi::Function::New(env, VoiceActivityDetectorDetectedWrapper));

  exports.Set(Napi::String::New(env, "voiceActivityDetectorPop"),
              Napi::Function::New(env, VoiceActivityDetectorPopWrapper));

  exports.Set(Napi::String::New(env, "voiceActivityDetectorClear"),
              Napi::Function::New(env, VoiceActivityDetectorClearWrapper));

  exports.Set(Napi::String::New(env, "voiceActivityDetectorFront"),
              Napi::Function::New(env, VoiceActivityDetectorFrontWrapper));

  exports.Set(Napi::String::New(env, "voiceActivityDetectorReset"),
              Napi::Function::New(env, VoiceActivityDetectorResetWrapper));
}
