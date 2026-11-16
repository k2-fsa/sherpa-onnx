// scripts/node-addon-api/src/forced-alignment.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include <memory>
#include <sstream>
#include <string>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxForcedAlignmentOptions GetForcedAlignmentOptions(
    Napi::Object obj) {
  SherpaOnnxForcedAlignmentOptions c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("options") || !obj.Get("options").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("options").As<Napi::Object>();

  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(beam, beam);

  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(retry_beam, retryBeam);

  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(transition_scale, transitionScale);

  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(self_loop_scale, selfLoopScale);

  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(acoustic_scale, acousticScale);

  return c;
}

static SherpaOnnxForcedAlignerConfig GetForcedAlignerConfig(
    Napi::Object obj) {
  SherpaOnnxForcedAlignerConfig c;
  memset(&c, 0, sizeof(c));

  Napi::Object o = obj;

  SHERPA_ONNX_ASSIGN_ATTR_STR(model_dir, modelDir);

  SHERPA_ONNX_ASSIGN_ATTR_STR(lexicon, lexicon);

  SHERPA_ONNX_ASSIGN_ATTR_BOOL(debug, debug);

  c.options = GetForcedAlignmentOptions(obj);

  return c;
}

static void FreeForcedAlignerConfig(const SherpaOnnxForcedAlignerConfig &c) {
  SHERPA_ONNX_DELETE_C_STR(c.model_dir);
  SHERPA_ONNX_DELETE_C_STR(c.lexicon);
}

static Napi::External<SherpaOnnxForcedAligner> CreateForcedAlignerWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
#if __OHOS__
  if (info.Length() != 1 && info.Length() != 2) {
    std::ostringstream os;
    os << "Expect 1 or 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }
#else
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }
#endif

  if (!info[0].IsObject()) {
    Napi::TypeError::New(env, "You should pass an object as the first argument.")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Object o = info[0].As<Napi::Object>();

  SherpaOnnxForcedAlignerConfig c = GetForcedAlignerConfig(o);

#if __OHOS__
  SherpaOnnxForcedAligner *aligner = nullptr;

  if (info.Length() == 1 || info[1].IsUndefined() || info[1].IsNull()) {
    aligner = SherpaOnnxCreateForcedAligner(&c);
  } else {
    if (!info[1].IsObject()) {
      Napi::TypeError::New(
          env, "You should pass a resource manager as the second argument.")
          .ThrowAsJavaScriptException();

      FreeForcedAlignerConfig(c);
      return {};
    }

    std::unique_ptr<NativeResourceManager,
                    decltype(&OH_ResourceManager_ReleaseNativeResourceManager)>
        mgr(OH_ResourceManager_InitNativeResourceManager(env, info[1]),
            &OH_ResourceManager_ReleaseNativeResourceManager);

    aligner = SherpaOnnxCreateForcedAlignerOHOS(&c, mgr.get());
  }
#else
  SherpaOnnxForcedAligner *aligner = SherpaOnnxCreateForcedAligner(&c);
#endif

  FreeForcedAlignerConfig(c);

  if (!aligner) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxForcedAligner>::New(
      env, aligner, [](Napi::Env /*env*/, SherpaOnnxForcedAligner *aligner) {
        SherpaOnnxDestroyForcedAligner(aligner);
      });
}

class CreateForcedAlignerAsyncWorker : public Napi::AsyncWorker {
 public:
  CreateForcedAlignerAsyncWorker(const Napi::Env &env,
                                 const SherpaOnnxForcedAlignerConfig &cfg,
                                 const Napi::Promise::Deferred &deferred)
      : Napi::AsyncWorker(env), cfg_(cfg), deferred_(deferred) {}

  void Execute() override {
    aligner_ = SherpaOnnxCreateForcedAligner(&cfg_);
    FreeForcedAlignerConfig(cfg_);

    if (!aligner_) {
      SetError("Failed to create the forced aligner");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();

    deferred_.Resolve(Napi::External<SherpaOnnxForcedAligner>::New(
        env, const_cast<SherpaOnnxForcedAligner *>(aligner_),
        [](Napi::Env /*env*/, SherpaOnnxForcedAligner *aligner) {
          SherpaOnnxDestroyForcedAligner(aligner);
        }));
  }

  void OnError(const Napi::Error &e) override { deferred_.Reject(e.Value()); }

 private:
  SherpaOnnxForcedAlignerConfig cfg_;
  const SherpaOnnxForcedAligner *aligner_ = nullptr;
  Napi::Promise::Deferred deferred_;
};

static Napi::Value CreateForcedAlignerAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "Expected config object")
        .ThrowAsJavaScriptException();

    return env.Null();
  }

  SherpaOnnxForcedAlignerConfig c =
      GetForcedAlignerConfig(info[0].As<Napi::Object>());

  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);

  auto *worker = new CreateForcedAlignerAsyncWorker(env, c, deferred);
  worker->Queue();

  return deferred.Promise();
}

static Napi::External<SherpaOnnxOfflineStream> ForcedAlignerCreateStreamWrapper(
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
        "You should pass a forced aligner pointer as the first argument")
        .ThrowAsJavaScriptException();

    return {};
  }

  SherpaOnnxForcedAligner *aligner =
      info[0].As<Napi::External<SherpaOnnxForcedAligner>>().Data();

  SherpaOnnxOfflineStream *stream =
      SherpaOnnxForcedAlignerCreateStream(aligner);

  return Napi::External<SherpaOnnxOfflineStream>::New(
      env, stream, [](Napi::Env /*env*/, SherpaOnnxOfflineStream *stream) {
        SherpaOnnxDestroyOfflineStream(stream);
      });
}

static Napi::Object GetForcedAlignmentResultObject(
    Napi::Env env, const SherpaOnnxForcedAlignmentResult *r) {
  Napi::Object ans = Napi::Object::New(env);

  if (!r) {
    ans.Set(Napi::String::New(env, "duration"), Napi::Number::New(env, 0));
    ans.Set(Napi::String::New(env, "textgrid"), Napi::String::New(env, ""));
    ans.Set(Napi::String::New(env, "tiers"), Napi::Array::New(env, 0));
    return ans;
  }

  ans.Set(Napi::String::New(env, "duration"),
          Napi::Number::New(env, r->duration));
  ans.Set(Napi::String::New(env, "textgrid"),
          Napi::String::New(env, r->textgrid ? r->textgrid : ""));

  Napi::Array tiers = Napi::Array::New(env, r->num_tiers);
  for (int32_t i = 0; i != r->num_tiers; ++i) {
    const SherpaOnnxForcedAlignmentTier &t = r->tiers[i];

    Napi::Object tier = Napi::Object::New(env);
    tier.Set(Napi::String::New(env, "name"),
             Napi::String::New(env, t.name ? t.name : ""));

    Napi::Array intervals = Napi::Array::New(env, t.num_intervals);
    for (int32_t k = 0; k != t.num_intervals; ++k) {
      const SherpaOnnxForcedAlignmentInterval &iv = t.intervals[k];

      Napi::Object interval = Napi::Object::New(env);
      interval.Set(Napi::String::New(env, "start"),
                   Napi::Number::New(env, iv.start));
      interval.Set(Napi::String::New(env, "end"),
                   Napi::Number::New(env, iv.end));
      interval.Set(Napi::String::New(env, "text"),
                   Napi::String::New(env, iv.text ? iv.text : ""));

      intervals.Set(k, interval);
    }
    tier.Set(Napi::String::New(env, "intervals"), intervals);

    tiers.Set(i, tier);
  }
  ans.Set(Napi::String::New(env, "tiers"), tiers);

  return ans;
}

static Napi::Object ForcedAlignerComputeWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 3) {
    std::ostringstream os;
    os << "Expect only 3 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return Napi::Object::New(env);
  }

  if (!info[0].IsExternal() || !info[1].IsExternal() || !info[2].IsString()) {
    Napi::TypeError::New(env,
                         "You should pass (forced aligner pointer, offline "
                         "stream pointer, text) as the arguments")
        .ThrowAsJavaScriptException();

    return Napi::Object::New(env);
  }

  SherpaOnnxForcedAligner *aligner =
      info[0].As<Napi::External<SherpaOnnxForcedAligner>>().Data();
  SherpaOnnxOfflineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOfflineStream>>().Data();

  Napi::String _str = info[2].As<Napi::String>();
  std::string text = _str.Utf8Value();

  const SherpaOnnxForcedAlignmentResult *r =
      SherpaOnnxForcedAlignerCompute(aligner, stream, text.c_str());

  Napi::Object ans = GetForcedAlignmentResultObject(env, r);

  SherpaOnnxDestroyForcedAlignmentResult(r);

  return ans;
}

class ForcedAlignerComputeAsyncWorker : public Napi::AsyncWorker {
 public:
  ForcedAlignerComputeAsyncWorker(Napi::Env env,
                                  const SherpaOnnxForcedAligner *aligner,
                                  const SherpaOnnxOfflineStream *stream,
                                  const std::string &text,
                                  Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        aligner_(aligner),
        stream_(stream),
        text_(text),
        deferred_(deferred) {}

  void Execute() override {
    result_ = SherpaOnnxForcedAlignerCompute(aligner_, stream_, text_.c_str());

    if (!result_) {
      SetError("Forced alignment failed");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();

    Napi::Object ans = GetForcedAlignmentResultObject(env, result_);

    SherpaOnnxDestroyForcedAlignmentResult(result_);

    deferred_.Resolve(ans);
  }

  void OnError(const Napi::Error &e) override { deferred_.Reject(e.Value()); }

 private:
  const SherpaOnnxForcedAligner *aligner_;
  const SherpaOnnxOfflineStream *stream_;
  std::string text_;
  const SherpaOnnxForcedAlignmentResult *result_ = nullptr;
  Napi::Promise::Deferred deferred_;
};

static Napi::Value ForcedAlignerComputeAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 3) {
    std::ostringstream os;
    os << "Expect only 3 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return env.Null();
  }

  if (!info[0].IsExternal() || !info[1].IsExternal() || !info[2].IsString()) {
    Napi::TypeError::New(env,
                         "You should pass (forced aligner pointer, offline "
                         "stream pointer, text) as the arguments")
        .ThrowAsJavaScriptException();

    return env.Null();
  }

  SherpaOnnxForcedAligner *aligner =
      info[0].As<Napi::External<SherpaOnnxForcedAligner>>().Data();
  SherpaOnnxOfflineStream *stream =
      info[1].As<Napi::External<SherpaOnnxOfflineStream>>().Data();

  Napi::String _str = info[2].As<Napi::String>();
  std::string text = _str.Utf8Value();

  Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);

  auto *worker =
      new ForcedAlignerComputeAsyncWorker(env, aligner, stream, text, deferred);
  worker->Queue();

  return deferred.Promise();
}

void InitForcedAligner(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createForcedAligner"),
              Napi::Function::New(env, CreateForcedAlignerWrapper));

  exports.Set(Napi::String::New(env, "createForcedAlignerAsync"),
              Napi::Function::New(env, CreateForcedAlignerAsyncWrapper));

  exports.Set(Napi::String::New(env, "forcedAlignerCreateStream"),
              Napi::Function::New(env, ForcedAlignerCreateStreamWrapper));

  exports.Set(Napi::String::New(env, "forcedAlignerCompute"),
              Napi::Function::New(env, ForcedAlignerComputeWrapper));

  exports.Set(Napi::String::New(env, "forcedAlignerComputeAsync"),
              Napi::Function::New(env, ForcedAlignerComputeAsyncWrapper));
}
