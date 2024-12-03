// scripts/node-addon-api/src/non-streaming-tts.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <algorithm>
#include <sstream>

#include "macros.h" // NOLINT
#include "napi.h"   // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxOfflineTtsVitsModelConfig GetOfflineTtsVitsModelConfig(Napi::Object obj) {
    SherpaOnnxOfflineTtsVitsModelConfig c;
    memset(&c, 0, sizeof(c));

    if (!obj.Has("vits") || !obj.Get("vits").IsObject()) {
        return c;
    }

    Napi::Object o = obj.Get("vits").As<Napi::Object>();
    SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
    SHERPA_ONNX_ASSIGN_ATTR_STR(lexicon, lexicon);
    SHERPA_ONNX_ASSIGN_ATTR_STR(tokens, tokens);
    SHERPA_ONNX_ASSIGN_ATTR_STR(data_dir, dataDir);
    SHERPA_ONNX_ASSIGN_ATTR_FLOAT(noise_scale, noiseScale);
    SHERPA_ONNX_ASSIGN_ATTR_FLOAT(noise_scale_w, noiseScaleW);
    SHERPA_ONNX_ASSIGN_ATTR_FLOAT(length_scale, lengthScale);
    SHERPA_ONNX_ASSIGN_ATTR_STR(dict_dir, dictDir);

    return c;
}

static SherpaOnnxOfflineTtsModelConfig GetOfflineTtsModelConfig(Napi::Object obj) {
    SherpaOnnxOfflineTtsModelConfig c;
    memset(&c, 0, sizeof(c));

    if (!obj.Has("model") || !obj.Get("model").IsObject()) {
        return c;
    }

    Napi::Object o = obj.Get("model").As<Napi::Object>();

    c.vits = GetOfflineTtsVitsModelConfig(o);

    SHERPA_ONNX_ASSIGN_ATTR_INT32(num_threads, numThreads);

    if (o.Has("debug") && (o.Get("debug").IsNumber() || o.Get("debug").IsBoolean())) {
        if (o.Get("debug").IsBoolean()) {
            c.debug = o.Get("debug").As<Napi::Boolean>().Value();
        } else {
            c.debug = o.Get("debug").As<Napi::Number>().Int32Value();
        }
    }

    SHERPA_ONNX_ASSIGN_ATTR_STR(provider, provider);

    return c;
}

static Napi::External<SherpaOnnxOfflineTts> CreateOfflineTtsWrapper(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();
#if __OHOS__
    // the last argument is the NativeResourceManager
    if (info.Length() != 2) {
        std::ostringstream os;
        os << "Expect only 2 arguments. Given: " << info.Length();

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
        Napi::TypeError::New(env, "Expect an object as the argument").ThrowAsJavaScriptException();

        return {};
    }

    Napi::Object o = info[0].As<Napi::Object>();

    SherpaOnnxOfflineTtsConfig c;
    memset(&c, 0, sizeof(c));

    c.model = GetOfflineTtsModelConfig(o);

    SHERPA_ONNX_ASSIGN_ATTR_STR(rule_fsts, ruleFsts);
    SHERPA_ONNX_ASSIGN_ATTR_INT32(max_num_sentences, maxNumSentences);
    SHERPA_ONNX_ASSIGN_ATTR_STR(rule_fars, ruleFars);

#if __OHOS__
    std::unique_ptr<NativeResourceManager, decltype(&OH_ResourceManager_ReleaseNativeResourceManager)> mgr(
        OH_ResourceManager_InitNativeResourceManager(env, info[1]), &OH_ResourceManager_ReleaseNativeResourceManager);
    SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTtsOHOS(&c, mgr.get());
#else
    SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&c);
#endif

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
        Napi::TypeError::New(env, "Please check your config!").ThrowAsJavaScriptException();

        return {};
    }

    return Napi::External<SherpaOnnxOfflineTts>::New(
        env, tts, [](Napi::Env env, SherpaOnnxOfflineTts *tts) { SherpaOnnxDestroyOfflineTts(tts); });
}

static Napi::Number OfflineTtsSampleRateWrapper(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if (info.Length() != 1) {
        std::ostringstream os;
        os << "Expect only 1 argument. Given: " << info.Length();

        Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

        return {};
    }

    if (!info[0].IsExternal()) {
        Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.").ThrowAsJavaScriptException();

        return {};
    }

    SherpaOnnxOfflineTts *tts = info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

    int32_t sample_rate = SherpaOnnxOfflineTtsSampleRate(tts);

    return Napi::Number::New(env, sample_rate);
}

static Napi::Number OfflineTtsNumSpeakersWrapper(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if (info.Length() != 1) {
        std::ostringstream os;
        os << "Expect only 1 argument. Given: " << info.Length();

        Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

        return {};
    }

    if (!info[0].IsExternal()) {
        Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.").ThrowAsJavaScriptException();

        return {};
    }

    SherpaOnnxOfflineTts *tts = info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

    int32_t num_speakers = SherpaOnnxOfflineTtsNumSpeakers(tts);

    return Napi::Number::New(env, num_speakers);
}

// synchronous version
static Napi::Object OfflineTtsGenerateWrapper(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if (info.Length() != 2) {
        std::ostringstream os;
        os << "Expect only 2 arguments. Given: " << info.Length();

        Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

        return {};
    }

    if (!info[0].IsExternal()) {
        Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.").ThrowAsJavaScriptException();

        return {};
    }

    SherpaOnnxOfflineTts *tts = info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

    if (!info[1].IsObject()) {
        Napi::TypeError::New(env, "Argument 1 should be an object").ThrowAsJavaScriptException();

        return {};
    }

    Napi::Object obj = info[1].As<Napi::Object>();

    if (!obj.Has("text")) {
        Napi::TypeError::New(env, "The argument object should have a field text").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Get("text").IsString()) {
        Napi::TypeError::New(env, "The object['text'] should be a string").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Has("sid")) {
        Napi::TypeError::New(env, "The argument object should have a field sid").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Get("sid").IsNumber()) {
        Napi::TypeError::New(env, "The object['sid'] should be a number").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Has("speed")) {
        Napi::TypeError::New(env, "The argument object should have a field speed").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Get("speed").IsNumber()) {
        Napi::TypeError::New(env, "The object['speed'] should be a number").ThrowAsJavaScriptException();

        return {};
    }

    bool enable_external_buffer = true;
    if (obj.Has("enableExternalBuffer") && obj.Get("enableExternalBuffer").IsBoolean()) {
        enable_external_buffer = obj.Get("enableExternalBuffer").As<Napi::Boolean>().Value();
    }

    Napi::String _text = obj.Get("text").As<Napi::String>();
    std::string text = _text.Utf8Value();
    int32_t sid = obj.Get("sid").As<Napi::Number>().Int32Value();
    float speed = obj.Get("speed").As<Napi::Number>().FloatValue();

    const SherpaOnnxGeneratedAudio *audio;
    if (obj.Has("callback") && obj.Get("callback").IsFunction()) {
        Napi::Function cb = obj.Get("callback").As<Napi::Function>();
        struct MyCallbackArg {
            Napi::Env *penv;
            Napi::Function *pcb;
        };

        MyCallbackArg arg = {&env, &cb};

        auto callback = [](const float *samples, int32_t n, void *arg) -> int {
            auto parg = reinterpret_cast<MyCallbackArg *>(arg);

            Napi::Float32Array float32Array = Napi::Float32Array::New(*parg->penv, n);

            std::copy(samples, samples + n, float32Array.Data());

            parg->pcb->Call({float32Array});
            return 1;
        };
        audio = SherpaOnnxOfflineTtsGenerateWithCallbackWithArg(tts, text.c_str(), sid, speed, callback, &arg);
    } else {
        audio = SherpaOnnxOfflineTtsGenerate(tts, text.c_str(), sid, speed);
    }

    if (enable_external_buffer) {
        Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
            env, const_cast<float *>(audio->samples), sizeof(float) * audio->n,
            [](Napi::Env /*env*/, void * /*data*/, const SherpaOnnxGeneratedAudio *hint) {
                SherpaOnnxDestroyOfflineTtsGeneratedAudio(hint);
            },
            audio);
        Napi::Float32Array float32Array = Napi::Float32Array::New(env, audio->n, arrayBuffer, 0);

        Napi::Object ans = Napi::Object::New(env);
        ans.Set(Napi::String::New(env, "samples"), float32Array);
        ans.Set(Napi::String::New(env, "sampleRate"), audio->sample_rate);
        return ans;
    } else {
        // don't use external buffer
        Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(env, sizeof(float) * audio->n);

        Napi::Float32Array float32Array = Napi::Float32Array::New(env, audio->n, arrayBuffer, 0);

        std::copy(audio->samples, audio->samples + audio->n, float32Array.Data());

        Napi::Object ans = Napi::Object::New(env);
        ans.Set(Napi::String::New(env, "samples"), float32Array);
        ans.Set(Napi::String::New(env, "sampleRate"), audio->sample_rate);
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
        return ans;
    }
}

class TtsGenerateWorker : public Napi::AsyncWorker {
public:
    TtsGenerateWorker(const Napi::Env &env, SherpaOnnxOfflineTts *tts, const std::string &text, float speed,
                      int32_t sid, bool use_external_buffer)
        : Napi::AsyncWorker{env, "TtsGenerateWorker"}, deferred_(env), tts_(tts), text_(text), speed_(speed),
          sid_(sid), use_external_buffer_(use_external_buffer) {}

    Napi::Promise Promise() { return deferred_.Promise(); }

protected:
    void Execute() override {
        audio_ = SherpaOnnxOfflineTtsGenerate(tts_, text_.c_str(), sid_, speed_);

    }

     void OnOK() override {
        
        Napi::Env env = deferred_.Env(); 
        Napi::Object ans = Napi::Object::New(env);
    if (use_external_buffer_) {
        Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
            env, const_cast<float *>(audio_->samples), sizeof(float) * audio_->n,
            [](Napi::Env /*env*/, void * /*data*/, const SherpaOnnxGeneratedAudio *hint) {
                SherpaOnnxDestroyOfflineTtsGeneratedAudio(hint);
            },
            audio_);
        Napi::Float32Array float32Array = Napi::Float32Array::New(env, audio_->n, arrayBuffer, 0);

        ans.Set(Napi::String::New(env, "samples"), float32Array);
        ans.Set(Napi::String::New(env, "sampleRate"), audio_->sample_rate);
    } else {
        // don't use external buffer
        Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(env, sizeof(float) * audio_->n);

        Napi::Float32Array float32Array = Napi::Float32Array::New(env, audio_->n, arrayBuffer, 0);

        std::copy(audio_->samples, audio_->samples + audio_->n, float32Array.Data());

        ans.Set(Napi::String::New(env, "samples"), float32Array);
        ans.Set(Napi::String::New(env, "sampleRate"), audio_->sample_rate);
        SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_);
    }
        
        deferred_.Resolve(ans);
    }

private:
    Napi::Promise::Deferred deferred_;
    SherpaOnnxOfflineTts *tts_;
    std::string text_;
    float speed_;
    int32_t sid_;
    bool use_external_buffer_;
    
    const SherpaOnnxGeneratedAudio *audio_;
};

static Napi::Object OfflineTtsGenerateAsyncWrapper(const Napi::CallbackInfo &info) {
    Napi::Env env = info.Env();

    if (info.Length() != 2) {
        std::ostringstream os;
        os << "Expect only 2 arguments. Given: " << info.Length();

        Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

        return {};
    }

    if (!info[0].IsExternal()) {
        Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.").ThrowAsJavaScriptException();

        return {};
    }

    SherpaOnnxOfflineTts *tts = info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

    if (!info[1].IsObject()) {
        Napi::TypeError::New(env, "Argument 1 should be an object").ThrowAsJavaScriptException();

        return {};
    }

    Napi::Object obj = info[1].As<Napi::Object>();

    if (!obj.Has("text")) {
        Napi::TypeError::New(env, "The argument object should have a field text").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Get("text").IsString()) {
        Napi::TypeError::New(env, "The object['text'] should be a string").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Has("sid")) {
        Napi::TypeError::New(env, "The argument object should have a field sid").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Get("sid").IsNumber()) {
        Napi::TypeError::New(env, "The object['sid'] should be a number").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Has("speed")) {
        Napi::TypeError::New(env, "The argument object should have a field speed").ThrowAsJavaScriptException();

        return {};
    }

    if (!obj.Get("speed").IsNumber()) {
        Napi::TypeError::New(env, "The object['speed'] should be a number").ThrowAsJavaScriptException();

        return {};
    }

    bool enable_external_buffer = true;
    if (obj.Has("enableExternalBuffer") && obj.Get("enableExternalBuffer").IsBoolean()) {
        enable_external_buffer = obj.Get("enableExternalBuffer").As<Napi::Boolean>().Value();
    }

    Napi::String _text = obj.Get("text").As<Napi::String>();
    std::string text = _text.Utf8Value();
    int32_t sid = obj.Get("sid").As<Napi::Number>().Int32Value();
    float speed = obj.Get("speed").As<Napi::Number>().FloatValue();

    const SherpaOnnxGeneratedAudio *audio;
     TtsGenerateWorker* worker = new TtsGenerateWorker(env, tts, text, speed, sid, enable_external_buffer);
    worker->Queue();
    return worker->Promise();
}

void InitNonStreamingTts(Napi::Env env, Napi::Object exports) {
    exports.Set(Napi::String::New(env, "createOfflineTts"), Napi::Function::New(env, CreateOfflineTtsWrapper));

    exports.Set(Napi::String::New(env, "getOfflineTtsSampleRate"),
                Napi::Function::New(env, OfflineTtsSampleRateWrapper));

    exports.Set(Napi::String::New(env, "getOfflineTtsNumSpeakers"),
                Napi::Function::New(env, OfflineTtsNumSpeakersWrapper));

    exports.Set(Napi::String::New(env, "offlineTtsGenerate"), Napi::Function::New(env, OfflineTtsGenerateWrapper));

    exports.Set(Napi::String::New(env, "offlineTtsGenerateAsync"),
                Napi::Function::New(env, OfflineTtsGenerateAsyncWrapper));
}
