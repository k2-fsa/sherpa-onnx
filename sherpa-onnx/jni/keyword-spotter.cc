// sherpa-onnx/jni/keyword-spotter.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/keyword-spotter.h"

#include <memory>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

OnlineModelConfig GetOnlineModelConfig(JNIEnv *env, jclass model_config_cls,
                                       jobject model_config, bool *ok);

static KeywordSpotterConfig GetKwsConfig(JNIEnv *env, jobject config,
                                         bool *ok) {
  KeywordSpotterConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  // https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html
  // https://courses.cs.washington.edu/courses/cse341/99wi/java/tutorial/native1.1/implementing/field.html

  //---------- decoding ----------
  SHERPA_ONNX_JNI_READ_INT(ans.max_active_paths, maxActivePaths, cls, config);

  SHERPA_ONNX_JNI_READ_STRING(ans.keywords_file, keywordsFile, cls, config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.keywords_score, keywordsScore, cls, config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.keywords_threshold, keywordsThreshold, cls,
                             config);

  SHERPA_ONNX_JNI_READ_INT(ans.num_trailing_blanks, numTrailingBlanks, cls,
                           config);

  //---------- feat config ----------
  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpa/onnx/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  SHERPA_ONNX_JNI_READ_INT(ans.feat_config.sampling_rate, sampleRate,
                           feat_config_cls, feat_config);

  SHERPA_ONNX_JNI_READ_INT(ans.feat_config.feature_dim, featureDim,
                           feat_config_cls, feat_config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.feat_config.dither, dither, feat_config_cls,
                             feat_config);

  //---------- model config ----------
  fid = env->GetFieldID(cls, "modelConfig",
                        "Lcom/k2fsa/sherpa/onnx/OnlineModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);
  ans.model_config =
      GetOnlineModelConfig(env, model_config_cls, model_config, ok);

  if (!*ok) {
    return ans;
  }

  // *ok = false;
  // If there are more fields, remember to set *ok to false

  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_KeywordSpotter_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  bool ok = false;
  auto config = sherpa_onnx::GetKwsConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto kws = new sherpa_onnx::KeywordSpotter(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)kws;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_KeywordSpotter_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  bool ok = false;
  auto config = sherpa_onnx::GetKwsConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto kws = new sherpa_onnx::KeywordSpotter(config);

  return (jlong)kws;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_KeywordSpotter_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::KeywordSpotter *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_KeywordSpotter_decode(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto kws = reinterpret_cast<sherpa_onnx::KeywordSpotter *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(stream_ptr);

  kws->DecodeStream(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_KeywordSpotter_reset(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto kws = reinterpret_cast<sherpa_onnx::KeywordSpotter *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(stream_ptr);

  kws->Reset(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_KeywordSpotter_createStream(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring keywords) {
  auto kws = reinterpret_cast<sherpa_onnx::KeywordSpotter *>(ptr);

  const char *p = env->GetStringUTFChars(keywords, nullptr);
  std::unique_ptr<sherpa_onnx::OnlineStream> stream;

  if (strlen(p) == 0) {
    stream = kws->CreateStream();
  } else {
    stream = kws->CreateStream(p);
  }

  env->ReleaseStringUTFChars(keywords, p);

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_onnx_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_onnx::OnlineStream *ans = stream.release();
  return (jlong)ans;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_KeywordSpotter_isReady(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto kws = reinterpret_cast<sherpa_onnx::KeywordSpotter *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(stream_ptr);

  return kws->IsReady(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobject JNICALL Java_com_k2fsa_sherpa_onnx_KeywordSpotter_getResult(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  auto kws = reinterpret_cast<sherpa_onnx::KeywordSpotter *>(ptr);
  auto stream = reinterpret_cast<sherpa_onnx::OnlineStream *>(stream_ptr);

  sherpa_onnx::KeywordResult result = kws->GetResult(stream);

  jstring j_keyword = env->NewStringUTF(result.keyword.c_str());

  // Convert tokens (std::vector<std::string> -> String[])
  jclass string_cls = env->FindClass("java/lang/String");
  if (string_cls == nullptr) {
    SHERPA_ONNX_LOGE("Failed to find class java/lang/String");
    return nullptr;
  }

  jobjectArray j_tokens =
      env->NewObjectArray(result.tokens.size(), string_cls, nullptr);

  for (size_t i = 0; i < result.tokens.size(); ++i) {
    jstring t = env->NewStringUTF(result.tokens[i].c_str());
    env->SetObjectArrayElement(j_tokens, i, t);
    env->DeleteLocalRef(t);
  }

  // Convert timestamps (std::vector<float> -> float[])
  jfloatArray j_timestamps = env->NewFloatArray(result.timestamps.size());
  env->SetFloatArrayRegion(j_timestamps, 0, result.timestamps.size(),
                           result.timestamps.data());

  // Find KeywordSpotterResult class
  jclass result_cls =
      env->FindClass("com/k2fsa/sherpa/onnx/KeywordSpotterResult");

  if (result_cls == nullptr) {
    SHERPA_ONNX_LOGE(
        "Failed to find class com/k2fsa/sherpa/onnx/KeywordSpotterResult");
    return nullptr;
  }

  jmethodID ctor = env->GetMethodID(
      result_cls, "<init>", "(Ljava/lang/String;[Ljava/lang/String;[F)V");

  if (ctor == nullptr) {
    SHERPA_ONNX_LOGE("Failed to get KeywordSpotterResult constructor");
    return nullptr;
  }

  // Create the KeywordSpotterResult object
  jobject result_obj =
      env->NewObject(result_cls, ctor, j_keyword, j_tokens, j_timestamps);

  env->DeleteLocalRef(j_keyword);
  env->DeleteLocalRef(j_tokens);
  env->DeleteLocalRef(j_timestamps);
  env->DeleteLocalRef(result_cls);
  env->DeleteLocalRef(string_cls);

  return result_obj;
}
