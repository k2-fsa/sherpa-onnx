// sherpa-onnx/jni/audio-tagging.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging.h"

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static AudioTaggingConfig GetAudioTaggingConfig(JNIEnv *env, jobject config,
                                                bool *ok) {
  AudioTaggingConfig ans;

  jclass cls = env->GetObjectClass(config);

  jfieldID fid = env->GetFieldID(
      cls, "model", "Lcom/k2fsa/sherpa/onnx/AudioTaggingModelConfig;");
  jobject model = env->GetObjectField(config, fid);
  jclass model_cls = env->GetObjectClass(model);

  fid = env->GetFieldID(
      model_cls, "zipformer",
      "Lcom/k2fsa/sherpa/onnx/OfflineZipformerAudioTaggingModelConfig;");
  jobject zipformer = env->GetObjectField(model, fid);
  jclass zipformer_cls = env->GetObjectClass(zipformer);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.zipformer.model, model, zipformer_cls,
                              zipformer);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.ced, ced, model_cls, model);

  SHERPA_ONNX_JNI_READ_INT(ans.model.num_threads, numThreads, model_cls, model);

  SHERPA_ONNX_JNI_READ_BOOL(ans.model.debug, debug, model_cls, model);

  SHERPA_ONNX_JNI_READ_STRING(ans.model.provider, provider, model_cls, model);

  SHERPA_ONNX_JNI_READ_STRING(ans.labels, labels, cls, config);

  SHERPA_ONNX_JNI_READ_INT(ans.top_k, topK, cls, config);

  *ok = true;
  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_AudioTagging_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif

  bool ok = false;
  auto config = sherpa_onnx::GetAudioTaggingConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("audio tagging newFromAsset config:\n%s",
                   config.ToString().c_str());

  auto tagger = new sherpa_onnx::AudioTagging(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)tagger;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_AudioTagging_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  bool ok = false;

  auto config = sherpa_onnx::GetAudioTaggingConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("audio tagging newFromFile config:\n%s",
                   config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto tagger = new sherpa_onnx::AudioTagging(config);

  return (jlong)tagger;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_AudioTagging_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::AudioTagging *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_AudioTagging_createStream(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto tagger = reinterpret_cast<sherpa_onnx::AudioTagging *>(ptr);
  std::unique_ptr<sherpa_onnx::OfflineStream> s = tagger->CreateStream();

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_onnx_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_onnx::OfflineStream *p = s.release();
  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL Java_com_k2fsa_sherpa_onnx_AudioTagging_compute(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong streamPtr, jint top_k) {
  auto tagger = reinterpret_cast<sherpa_onnx::AudioTagging *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OfflineStream *>(streamPtr);
  std::vector<sherpa_onnx::AudioEvent> events = tagger->Compute(stream, top_k);

  // Find the AudioEvent class
  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/AudioEvent");
  if (cls == nullptr) {
    SHERPA_ONNX_LOGE("Failed to find class com/k2fsa/sherpa/onnx/AudioEvent");
    return nullptr;
  }

  // Get the constructor: AudioEvent(String name, int index, float prob)
  jmethodID ctor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;IF)V");
  if (ctor == nullptr) {
    SHERPA_ONNX_LOGE("Failed to get AudioEvent constructor");
    return nullptr;
  }

  // Create a jobjectArray of AudioEvent
  jobjectArray obj_arr = env->NewObjectArray(events.size(), cls, nullptr);

  for (size_t i = 0; i < events.size(); ++i) {
    const auto &e = events[i];

    jstring name = env->NewStringUTF(e.name.c_str());
    jobject event_obj = env->NewObject(cls, ctor, name, e.index, e.prob);

    env->SetObjectArrayElement(obj_arr, i, event_obj);

    env->DeleteLocalRef(name);
    env->DeleteLocalRef(event_obj);
  }

  env->DeleteLocalRef(cls);

  return obj_arr;
}
