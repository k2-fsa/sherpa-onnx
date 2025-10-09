// sherpa-onnx/jni/offline-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speaker-diarization.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static OfflineSpeakerDiarizationConfig GetOfflineSpeakerDiarizationConfig(
    JNIEnv *env, jobject config, bool *ok) {
  OfflineSpeakerDiarizationConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  //---------- segmentation ----------
  fid = env->GetFieldID(
      cls, "segmentation",
      "Lcom/k2fsa/sherpa/onnx/OfflineSpeakerSegmentationModelConfig;");
  jobject segmentation_config = env->GetObjectField(config, fid);
  jclass segmentation_config_cls = env->GetObjectClass(segmentation_config);

  fid = env->GetFieldID(
      segmentation_config_cls, "pyannote",
      "Lcom/k2fsa/sherpa/onnx/OfflineSpeakerSegmentationPyannoteModelConfig;");
  jobject pyannote_config = env->GetObjectField(segmentation_config, fid);
  jclass pyannote_config_cls = env->GetObjectClass(pyannote_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.segmentation.pyannote.model, model,
                              pyannote_config_cls, pyannote_config);

  SHERPA_ONNX_JNI_READ_INT(ans.segmentation.num_threads, numThreads,
                           segmentation_config_cls, segmentation_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.segmentation.debug, debug,
                            segmentation_config_cls, segmentation_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.segmentation.provider, provider,
                              segmentation_config_cls, segmentation_config);

  //---------- embedding ----------
  fid = env->GetFieldID(
      cls, "embedding",
      "Lcom/k2fsa/sherpa/onnx/SpeakerEmbeddingExtractorConfig;");
  jobject embedding_config = env->GetObjectField(config, fid);
  jclass embedding_config_cls = env->GetObjectClass(embedding_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.embedding.model, model, embedding_config_cls,
                              embedding_config);

  SHERPA_ONNX_JNI_READ_INT(ans.embedding.num_threads, numThreads,
                           embedding_config_cls, embedding_config);

  SHERPA_ONNX_JNI_READ_BOOL(ans.embedding.debug, debug, embedding_config_cls,
                            embedding_config);

  SHERPA_ONNX_JNI_READ_STRING(ans.embedding.provider, provider,
                              embedding_config_cls, embedding_config);

  fid = env->GetFieldID(cls, "clustering",
                        "Lcom/k2fsa/sherpa/onnx/FastClusteringConfig;");
  jobject clustering_config = env->GetObjectField(config, fid);
  jclass clustering_config_cls = env->GetObjectClass(clustering_config);

  SHERPA_ONNX_JNI_READ_INT(ans.clustering.num_clusters, numClusters,
                           clustering_config_cls, clustering_config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.clustering.threshold, threshold,
                             clustering_config_cls, clustering_config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.min_duration_on, minDurationOn, cls, config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.min_duration_off, minDurationOff, cls, config);

  *ok = true;
  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeakerDiarization_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif

  bool ok = false;
  auto config =
      sherpa_onnx::GetOfflineSpeakerDiarizationConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto sd = new sherpa_onnx::OfflineSpeakerDiarization(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)sd;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeakerDiarization_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  bool ok = false;
  auto config =
      sherpa_onnx::GetOfflineSpeakerDiarizationConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return 0;
  }

  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
    return 0;
  }

  auto sd = new sherpa_onnx::OfflineSpeakerDiarization(config);

  return (jlong)sd;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeakerDiarization_setConfig(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jobject _config) {
  bool ok = false;
  auto config =
      sherpa_onnx::GetOfflineSpeakerDiarizationConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Please read the error message carefully");
    return;
  }

  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());

  auto sd = reinterpret_cast<sherpa_onnx::OfflineSpeakerDiarization *>(ptr);
  sd->SetConfig(config);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeakerDiarization_delete(JNIEnv * /*env*/,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::OfflineSpeakerDiarization *>(ptr);
}

static jobjectArray ProcessImpl(
    JNIEnv *env,
    const std::vector<sherpa_onnx::OfflineSpeakerDiarizationSegment>
        &segments) {
  jclass cls =
      env->FindClass("com/k2fsa/sherpa/onnx/OfflineSpeakerDiarizationSegment");

  jobjectArray obj_arr =
      (jobjectArray)env->NewObjectArray(segments.size(), cls, nullptr);

  jmethodID constructor = env->GetMethodID(cls, "<init>", "(FFI)V");

  for (int32_t i = 0; i != segments.size(); ++i) {
    const auto &s = segments[i];
    jobject segment =
        env->NewObject(cls, constructor, s.Start(), s.End(), s.Speaker());
    env->SetObjectArrayElement(obj_arr, i, segment);
  }

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeakerDiarization_process(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples) {
  auto sd = reinterpret_cast<sherpa_onnx::OfflineSpeakerDiarization *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  auto segments = sd->Process(p, n).SortByStartTime();
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  return ProcessImpl(env, segments);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeakerDiarization_processWithCallback(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jobject callback, jlong arg) {
  std::function<int32_t(int32_t, int32_t, void *)> callback_wrapper =
      [env, callback](int32_t num_processed_chunks, int32_t num_total_chunks,
                      void *data) -> int {
    jclass cls = env->GetObjectClass(callback);

    jmethodID mid = env->GetMethodID(cls, "invoke", "(IIJ)Ljava/lang/Integer;");
    if (mid == nullptr) {
      SHERPA_ONNX_LOGE("Failed to get the callback. Ignore it.");
      return 0;
    }

    jobject ret = env->CallObjectMethod(callback, mid, num_processed_chunks,
                                        num_total_chunks, (jlong)data);
    jclass jklass = env->GetObjectClass(ret);
    jmethodID int_value_mid = env->GetMethodID(jklass, "intValue", "()I");
    return env->CallIntMethod(ret, int_value_mid);
  };

  auto sd = reinterpret_cast<sherpa_onnx::OfflineSpeakerDiarization *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  auto segments =
      sd->Process(p, n, callback_wrapper, reinterpret_cast<void *>(arg))
          .SortByStartTime();
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  return ProcessImpl(env, segments);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL
Java_com_k2fsa_sherpa_onnx_OfflineSpeakerDiarization_getSampleRate(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  return reinterpret_cast<sherpa_onnx::OfflineSpeakerDiarization *>(ptr)
      ->SampleRate();
}
