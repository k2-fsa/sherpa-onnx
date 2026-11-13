// sherpa-onnx/jni/forced-aligner.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/forced-aligner.h"

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static ForcedAlignmentOptions GetForcedAlignmentOptions(JNIEnv *env,
                                                        jobject options) {
  ForcedAlignmentOptions ans;

  jclass cls = env->GetObjectClass(options);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.beam, beam, cls, options);
  SHERPA_ONNX_JNI_READ_FLOAT(ans.retry_beam, retryBeam, cls, options);
  SHERPA_ONNX_JNI_READ_FLOAT(ans.transition_scale, transitionScale, cls,
                             options);
  SHERPA_ONNX_JNI_READ_FLOAT(ans.self_loop_scale, selfLoopScale, cls, options);
  SHERPA_ONNX_JNI_READ_FLOAT(ans.acoustic_scale, acousticScale, cls, options);

  return ans;
}

static ForcedAlignerConfig GetForcedAlignerConfig(JNIEnv *env, jobject config,
                                                  bool *ok) {
  ForcedAlignerConfig ans;

  jclass cls = env->GetObjectClass(config);

  SHERPA_ONNX_JNI_READ_STRING(ans.model_dir, modelDir, cls, config);
  SHERPA_ONNX_JNI_READ_STRING(ans.lexicon, lexicon, cls, config);
  SHERPA_ONNX_JNI_READ_BOOL(ans.debug, debug, cls, config);

  jfieldID fid = env->GetFieldID(
      cls, "options", "Lcom/k2fsa/sherpa/onnx/ForcedAlignmentOptions;");
  jobject options = env->GetObjectField(config, fid);
  ans.options = GetForcedAlignmentOptions(env, options);

  *ok = true;
  return ans;
}

static jobject NewAlignmentInterval(JNIEnv *env, const AlignmentInterval &iv) {
  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/AlignmentInterval");
  jmethodID ctor = env->GetMethodID(cls, "<init>", "(FFLjava/lang/String;)V");
  jstring text = env->NewStringUTF(iv.text.c_str());
  jobject ans = env->NewObject(cls, ctor, iv.start, iv.end, text);
  env->DeleteLocalRef(cls);
  env->DeleteLocalRef(text);
  return ans;
}

static jobject NewAlignmentTier(JNIEnv *env, const AlignmentTier &tier) {
  jclass interval_cls =
      env->FindClass("com/k2fsa/sherpa/onnx/AlignmentInterval");
  jobjectArray intervals =
      env->NewObjectArray(tier.intervals.size(), interval_cls, nullptr);
  for (size_t i = 0; i != tier.intervals.size(); ++i) {
    jobject interval = NewAlignmentInterval(env, tier.intervals[i]);
    env->SetObjectArrayElement(intervals, i, interval);
    env->DeleteLocalRef(interval);
  }
  env->DeleteLocalRef(interval_cls);

  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/AlignmentTier");
  jmethodID ctor = env->GetMethodID(
      cls, "<init>",
      "(Ljava/lang/String;[Lcom/k2fsa/sherpa/onnx/AlignmentInterval;)V");
  jstring name = env->NewStringUTF(tier.name.c_str());
  jobject ans = env->NewObject(cls, ctor, name, intervals);
  env->DeleteLocalRef(cls);
  env->DeleteLocalRef(name);
  env->DeleteLocalRef(intervals);
  return ans;
}

static jobject NewForcedAlignmentResult(JNIEnv *env,
                                        const ForcedAlignmentResult &result) {
  jclass tier_cls = env->FindClass("com/k2fsa/sherpa/onnx/AlignmentTier");
  jobjectArray tiers =
      env->NewObjectArray(result.tiers.size(), tier_cls, nullptr);
  for (size_t i = 0; i != result.tiers.size(); ++i) {
    jobject tier = NewAlignmentTier(env, result.tiers[i]);
    env->SetObjectArrayElement(tiers, i, tier);
    env->DeleteLocalRef(tier);
  }
  env->DeleteLocalRef(tier_cls);

  jclass cls = env->FindClass("com/k2fsa/sherpa/onnx/ForcedAlignmentResult");
  jmethodID ctor = env->GetMethodID(cls, "<init>",
                                    "(F[Lcom/k2fsa/sherpa/onnx/AlignmentTier;"
                                    "Ljava/lang/String;)V");
  std::string text_grid = result.ToTextGrid();
  jstring textgrid = env->NewStringUTF(text_grid.c_str());
  jobject ans = env->NewObject(cls, ctor, result.duration, tiers, textgrid);
  env->DeleteLocalRef(cls);
  env->DeleteLocalRef(textgrid);
  env->DeleteLocalRef(tiers);
  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_ForcedAligner_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  bool ok = false;
  auto config = sherpa_onnx::GetForcedAlignerConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  if (config.debug) {
    SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  }

  sherpa_onnx::ForcedAligner *aligner = nullptr;
  try {
    aligner = new sherpa_onnx::ForcedAligner(config);
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("Failed to create ForcedAligner: %s", e.what());
    delete aligner;
    return 0;
  }

  return (jlong)aligner;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_ForcedAligner_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  bool ok = false;
  auto config = sherpa_onnx::GetForcedAlignerConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  if (config.debug) {
    SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  }

  sherpa_onnx::ForcedAligner *aligner = nullptr;
  try {
    aligner = new sherpa_onnx::ForcedAligner(
#if __ANDROID_API__ >= 9
        mgr,
#endif
        config);
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("Failed to create ForcedAligner: %s", e.what());
    delete aligner;
    return 0;
  }

  return (jlong)aligner;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_ForcedAligner_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::ForcedAligner *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_ForcedAligner_createStream(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  auto aligner = reinterpret_cast<sherpa_onnx::ForcedAligner *>(ptr);
  std::unique_ptr<sherpa_onnx::OfflineStream> s = aligner->CreateStream();

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_onnx_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_onnx::OfflineStream *p = s.release();
  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL Java_com_k2fsa_sherpa_onnx_ForcedAligner_compute(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong stream_ptr, jstring text) {
  auto aligner = reinterpret_cast<sherpa_onnx::ForcedAligner *>(ptr);
  auto stream =
      reinterpret_cast<const sherpa_onnx::OfflineStream *>(stream_ptr);

  const char *p_text = env->GetStringUTFChars(text, nullptr);
  std::string cpp_text(p_text);
  env->ReleaseStringUTFChars(text, p_text);

  sherpa_onnx::ForcedAlignmentResult result =
      aligner->Compute(stream, cpp_text);

  return sherpa_onnx::NewForcedAlignmentResult(env, result);
}
