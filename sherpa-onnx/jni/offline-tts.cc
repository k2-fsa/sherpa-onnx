// sherpa-onnx/jni/offline-tts.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

class SherpaOnnxOfflineTts {
 public:
#if __ANDROID_API__ >= 9
  SherpaOnnxOfflineTts(AAssetManager *mgr, const OfflineTtsConfig &config)
      : tts_(mgr, config) {}
#endif
  explicit SherpaOnnxOfflineTts(const OfflineTtsConfig &config)
      : tts_(config) {}

  GeneratedAudio Generate(const std::string &text, int64_t sid = 0,
                          float speed = 1.0,
                          std::function<void(const float *, int32_t, float)>
                              callback = nullptr) const {
    return tts_.Generate(text, sid, speed, callback);
  }

  int32_t SampleRate() const { return tts_.SampleRate(); }

  int32_t NumSpeakers() const { return tts_.NumSpeakers(); }

 private:
  OfflineTts tts_;
};

static OfflineTtsConfig GetOfflineTtsConfig(JNIEnv *env, jobject config) {
  OfflineTtsConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  fid = env->GetFieldID(cls, "model",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTtsModelConfig;");
  jobject model = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model);

  fid = env->GetFieldID(model_config_cls, "vits",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTtsVitsModelConfig;");
  jobject vits = env->GetObjectField(model, fid);
  jclass vits_cls = env->GetObjectClass(vits);

  fid = env->GetFieldID(vits_cls, "model", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(vits, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.model.vits.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(vits_cls, "lexicon", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(vits, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.vits.lexicon = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(vits_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(vits, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.vits.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(vits_cls, "dataDir", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(vits, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.vits.data_dir = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(vits_cls, "noiseScale", "F");
  ans.model.vits.noise_scale = env->GetFloatField(vits, fid);

  fid = env->GetFieldID(vits_cls, "noiseScaleW", "F");
  ans.model.vits.noise_scale_w = env->GetFloatField(vits, fid);

  fid = env->GetFieldID(vits_cls, "lengthScale", "F");
  ans.model.vits.length_scale = env->GetFloatField(vits, fid);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");
  ans.model.num_threads = env->GetIntField(model, fid);

  fid = env->GetFieldID(model_config_cls, "debug", "Z");
  ans.model.debug = env->GetBooleanField(model, fid);

  fid = env->GetFieldID(model_config_cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model.provider = p;
  env->ReleaseStringUTFChars(s, p);

  // for ruleFsts
  fid = env->GetFieldID(cls, "ruleFsts", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.rule_fsts = p;
  env->ReleaseStringUTFChars(s, p);

  // for ruleFars
  fid = env->GetFieldID(cls, "ruleFars", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.rule_fars = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "maxNumSentences", "I");
  ans.max_num_sentences = env->GetIntField(config, fid);

  return ans;
}

}  // namespace sherpa_onnx

#if SHERPA_ONNX_ENABLE_TTS == 1
SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_new(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
  }
#endif
  auto config = sherpa_onnx::GetOfflineTtsConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto tts = new sherpa_onnx::SherpaOnnxOfflineTts(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)tts;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_onnx::GetOfflineTtsConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
  }

  auto tts = new sherpa_onnx::SherpaOnnxOfflineTts(config);

  return (jlong)tts;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::SherpaOnnxOfflineTts *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_getSampleRate(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_onnx::SherpaOnnxOfflineTts *>(ptr)
      ->SampleRate();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_onnx_OfflineTts_getNumSpeakers(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_onnx::SherpaOnnxOfflineTts *>(ptr)
      ->NumSpeakers();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineTts_generateImpl(JNIEnv *env, jobject /*obj*/,
                                                   jlong ptr, jstring text,
                                                   jint sid, jfloat speed) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);
  SHERPA_ONNX_LOGE("string is: %s", p_text);

  auto audio =
      reinterpret_cast<sherpa_onnx::SherpaOnnxOfflineTts *>(ptr)->Generate(
          p_text, sid, speed);

  jfloatArray samples_arr = env->NewFloatArray(audio.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, audio.samples.size(),
                           audio.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, audio.sample_rate));

  env->ReleaseStringUTFChars(text, p_text);

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineTts_generateWithCallbackImpl(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring text, jint sid,
    jfloat speed, jobject callback) {
  const char *p_text = env->GetStringUTFChars(text, nullptr);
  SHERPA_ONNX_LOGE("string is: %s", p_text);

  std::function<void(const float *, int32_t, float)> callback_wrapper =
      [env, callback](const float *samples, int32_t n, float /*p*/) {
        jclass cls = env->GetObjectClass(callback);
        jmethodID mid = env->GetMethodID(cls, "invoke", "([F)V");

        jfloatArray samples_arr = env->NewFloatArray(n);
        env->SetFloatArrayRegion(samples_arr, 0, n, samples);
        env->CallVoidMethod(callback, mid, samples_arr);
      };

  auto audio =
      reinterpret_cast<sherpa_onnx::SherpaOnnxOfflineTts *>(ptr)->Generate(
          p_text, sid, speed, callback_wrapper);

  jfloatArray samples_arr = env->NewFloatArray(audio.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, audio.samples.size(),
                           audio.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, audio.sample_rate));

  env->ReleaseStringUTFChars(text, p_text);

  return obj_arr;
}
#endif
