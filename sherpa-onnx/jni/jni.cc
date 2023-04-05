// sherpa-onnx/jni/jni.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation
//                2022       Pingfeng Luo
//                2023       Zhaoming

// TODO(fangjun): Add documentation to functions/methods in this file
// and also show how to use them with kotlin, possibly with java.

// If you use ndk, you can find "jni.h" inside
// android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
#include "jni.h"  // NOLINT

#include <strstream>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#else
#include <fstream>
#endif
#include <mutex>
#include <unordered_map>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/wave-reader.h"

#define SHERPA_ONNX_EXTERN_C extern "C"

namespace sherpa_onnx {

class SherpaOnnx {
  //  use  map to manage streams based on its uuid,
  // it can be used for parallel stream decode
 public:
  SherpaOnnx(
#if __ANDROID_API__ >= 9
      AAssetManager *mgr,
#endif
      const sherpa_onnx::OnlineRecognizerConfig &config)
      : recognizer_(
#if __ANDROID_API__ >= 9
            mgr,
#endif
            config) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::shared_ptr<OnlineStream> s = recognizer_.CreateStream();
    s_map.insert({0, std::move(s)});
  }

  void AcceptWaveform(int32_t sample_rate, const float *samples, int32_t n,
                      int32_t s_id) {
    if (input_sample_rate_ == -1) {
      input_sample_rate_ = sample_rate;
    }
    auto stream_ = findStream(s_id);
    stream_->AcceptWaveform(sample_rate, samples, n);
  }

  void InputFinished(int32_t s_id) const {
    std::shared_ptr<OnlineStream> stream_ = findStream(s_id);
    std::vector<float> tail_padding(input_sample_rate_ * 0.32, 0);
    stream_->AcceptWaveform(input_sample_rate_, tail_padding.data(),
                            tail_padding.size());
    stream_->InputFinished();
  }

  const std::string GetText(int32_t s_id) const {
    auto stream_ = findStream(s_id);
    auto result = recognizer_.GetResult(stream_.get());
    return result.text;
  }

  bool IsEndpoint(int32_t s_id) const {
    auto stream_ = findStream(s_id);
    return recognizer_.IsEndpoint(stream_.get());
  }

  bool IsReady(int32_t s_id) const {
    auto stream_ = findStream(s_id);
    return recognizer_.IsReady(stream_.get());
  }

  void Reset(int32_t s_id) const {
    auto stream_ = findStream(s_id);
    return recognizer_.Reset(stream_.get());
  }

  void Decode(int32_t s_id) const {
    auto stream_ = findStream(s_id);
    recognizer_.DecodeStream(stream_.get());
  }

  int32_t CreatStream() {
    // create a new stream for recognizer
    std::unique_lock<std::mutex> lock(mutex_);
    std::shared_ptr<OnlineStream> s = recognizer_.CreateStream();
    int32_t s_id = GenerateId();
    s_map.insert({s_id, std::move(s)});

    return s_id;
  }
  void DecodeStreams() const {
    // not implemented
  }
  void ReleaseStreams(int32_t s_id) {
    if (s_id == 0) {
      return;  // 0 is used for default single stream, can not be release
    }
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = s_map.find(s_id);
    if (iter != s_map.end()) {
      s_map.erase(iter);
    }
  }

 private:
  std::shared_ptr<OnlineStream> findStream(int32_t s_id) const {
    auto iter = s_map.find(s_id);

    if (iter != s_map.end())

      return iter->second;

    else
      SHERPA_ONNX_LOGE("Do not Find stream id %d \n", s_id);

    return nullptr;
  }
  int32_t GenerateId() {
    if (current_id >= std::numeric_limits<std::int32_t>::max()) {
      // when id reach max value, set it to 0
      current_id = 1;
    }
    return (current_id++);
  }

  sherpa_onnx::OnlineRecognizer recognizer_;
  std::unordered_map<int32_t, std::shared_ptr<OnlineStream>> s_map;
  int32_t input_sample_rate_ = -1;
  int32_t current_id = 1;  // keep 0 id for the first stream in construction
  std::mutex mutex_;
};

// based on GetConfig in jni.cc
static OnlineRecognizerConfig GetConfigJava(JNIEnv *env, jobject config) {
  OnlineRecognizerConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  // https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html
  // https://courses.cs.washington.edu/courses/cse341/99wi/java/tutorial/native1.1/implementing/field.html

  //---------- decoding ----------
  fid = env->GetFieldID(cls, "decodingMethod", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.decoding_method = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "maxActivePaths", "I");
  ans.max_active_paths = env->GetIntField(config, fid);

  //---------- feat config ----------
  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpaonnx/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  fid = env->GetFieldID(feat_config_cls, "sampleRate", "I");
  ans.feat_config.sampling_rate = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "featureDim", "I");
  ans.feat_config.feature_dim = env->GetIntField(feat_config, fid);

  //---------- enable endpoint ----------
  fid = env->GetFieldID(cls, "enableEndpoint", "Z");
  ans.enable_endpoint = env->GetBooleanField(config, fid);

  //---------- endpoint_config ----------

  fid = env->GetFieldID(cls, "endpointConfig",
                        "Lcom/k2fsa/sherpaonnx/EndpointConfig;");
  jobject endpoint_config = env->GetObjectField(config, fid);
  jclass endpoint_config_cls = env->GetObjectClass(endpoint_config);

  fid = env->GetFieldID(endpoint_config_cls, "rule1",
                        "Lcom/k2fsa/sherpaonnx/EndpointRule;");
  jobject rule1 = env->GetObjectField(endpoint_config, fid);
  jclass rule_class = env->GetObjectClass(rule1);

  fid = env->GetFieldID(endpoint_config_cls, "rule2",
                        "Lcom/k2fsa/sherpaonnx/EndpointRule;");
  jobject rule2 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(endpoint_config_cls, "rule3",
                        "Lcom/k2fsa/sherpaonnx/EndpointRule;");
  jobject rule3 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(rule_class, "mustContainNonSilence", "Z");
  ans.endpoint_config.rule1.must_contain_nonsilence =
      env->GetBooleanField(rule1, fid);
  ans.endpoint_config.rule2.must_contain_nonsilence =
      env->GetBooleanField(rule2, fid);
  ans.endpoint_config.rule3.must_contain_nonsilence =
      env->GetBooleanField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minTrailingSilence", "F");
  ans.endpoint_config.rule1.min_trailing_silence =
      env->GetFloatField(rule1, fid);
  ans.endpoint_config.rule2.min_trailing_silence =
      env->GetFloatField(rule2, fid);
  ans.endpoint_config.rule3.min_trailing_silence =
      env->GetFloatField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minUtteranceLength", "F");
  ans.endpoint_config.rule1.min_utterance_length =
      env->GetFloatField(rule1, fid);
  ans.endpoint_config.rule2.min_utterance_length =
      env->GetFloatField(rule2, fid);
  ans.endpoint_config.rule3.min_utterance_length =
      env->GetFloatField(rule3, fid);

  //---------- model config ----------
  fid = env->GetFieldID(cls, "modelConfig",
                        "Lcom/k2fsa/sherpaonnx/OnlineTransducerModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  fid = env->GetFieldID(model_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.encoder_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.decoder_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "joiner", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.joiner_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");
  ans.model_config.num_threads = env->GetIntField(model_config, fid);

  fid = env->GetFieldID(model_config_cls, "debug", "Z");
  ans.model_config.debug = env->GetBooleanField(model_config, fid);

  return ans;
}
static OnlineRecognizerConfig GetConfig(JNIEnv *env, jobject config) {
  OnlineRecognizerConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  // https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html
  // https://courses.cs.washington.edu/courses/cse341/99wi/java/tutorial/native1.1/implementing/field.html

  //---------- decoding ----------
  fid = env->GetFieldID(cls, "decodingMethod", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.decoding_method = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "maxActivePaths", "I");
  ans.max_active_paths = env->GetIntField(config, fid);

  //---------- feat config ----------
  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpa/onnx/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  fid = env->GetFieldID(feat_config_cls, "sampleRate", "I");
  ans.feat_config.sampling_rate = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "featureDim", "I");
  ans.feat_config.feature_dim = env->GetIntField(feat_config, fid);

  //---------- enable endpoint ----------
  fid = env->GetFieldID(cls, "enableEndpoint", "Z");
  ans.enable_endpoint = env->GetBooleanField(config, fid);

  //---------- endpoint_config ----------

  fid = env->GetFieldID(cls, "endpointConfig",
                        "Lcom/k2fsa/sherpa/onnx/EndpointConfig;");
  jobject endpoint_config = env->GetObjectField(config, fid);
  jclass endpoint_config_cls = env->GetObjectClass(endpoint_config);

  fid = env->GetFieldID(endpoint_config_cls, "rule1",
                        "Lcom/k2fsa/sherpa/onnx/EndpointRule;");
  jobject rule1 = env->GetObjectField(endpoint_config, fid);
  jclass rule_class = env->GetObjectClass(rule1);

  fid = env->GetFieldID(endpoint_config_cls, "rule2",
                        "Lcom/k2fsa/sherpa/onnx/EndpointRule;");
  jobject rule2 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(endpoint_config_cls, "rule3",
                        "Lcom/k2fsa/sherpa/onnx/EndpointRule;");
  jobject rule3 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(rule_class, "mustContainNonSilence", "Z");
  ans.endpoint_config.rule1.must_contain_nonsilence =
      env->GetBooleanField(rule1, fid);
  ans.endpoint_config.rule2.must_contain_nonsilence =
      env->GetBooleanField(rule2, fid);
  ans.endpoint_config.rule3.must_contain_nonsilence =
      env->GetBooleanField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minTrailingSilence", "F");
  ans.endpoint_config.rule1.min_trailing_silence =
      env->GetFloatField(rule1, fid);
  ans.endpoint_config.rule2.min_trailing_silence =
      env->GetFloatField(rule2, fid);
  ans.endpoint_config.rule3.min_trailing_silence =
      env->GetFloatField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minUtteranceLength", "F");
  ans.endpoint_config.rule1.min_utterance_length =
      env->GetFloatField(rule1, fid);
  ans.endpoint_config.rule2.min_utterance_length =
      env->GetFloatField(rule2, fid);
  ans.endpoint_config.rule3.min_utterance_length =
      env->GetFloatField(rule3, fid);

  //---------- model config ----------
  fid = env->GetFieldID(cls, "modelConfig",
                        "Lcom/k2fsa/sherpa/onnx/OnlineTransducerModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  fid = env->GetFieldID(model_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.encoder_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.decoder_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "joiner", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.joiner_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");
  ans.model_config.num_threads = env->GetIntField(model_config, fid);

  fid = env->GetFieldID(model_config_cls, "debug", "Z");
  ans.model_config.debug = env->GetBooleanField(model_config, fid);

  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_new(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
  }
#endif

  auto config = sherpa_onnx::GetConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::SherpaOnnx(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_reset(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  model->Reset(0);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_isReady(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  return model->IsReady(0);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_isEndpoint(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  return model->IsEndpoint(0);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_decode(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  model->Decode(0);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  model->AcceptWaveform(sample_rate, p, n, 0);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_inputFinished(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr)->InputFinished(0);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_getText(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  // see
  // https://stackoverflow.com/questions/11621449/send-c-string-to-java-via-jni
  auto text = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr)->GetText(0);
  return env->NewStringUTF(text.c_str());
}

// ********************************************************************
// all inferface are used for JNI in java
SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpaonnx_OnlineRecognizer_acceptWaveform(
    JNIEnv *env, jobject objptr, jlong ptr, jfloatArray samples,
    jint sample_rate, jint sid) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  model->AcceptWaveform(sample_rate, p, n, sid);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpaonnx_OnlineRecognizer_inputFinished(
    JNIEnv *env, jobject objptr, jlong ptr, jint sid) {
  reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr)->InputFinished(sid);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpaonnx_OnlineRecognizer_getText(
    JNIEnv *env, jobject objptr, jlong ptr, jint sid) {
  auto text = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr)->GetText(sid);
  return env->NewStringUTF(text.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpaonnx_OnlineRecognizer_reset(
    JNIEnv *env, jobject objptr, jlong ptr, jint sid) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);

  model->Reset(sid);

  // SHERPA_ONNX_LOGE("reset sid:\n%s", new_sid.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpaonnx_OnlineRecognizer_decode(
    JNIEnv *env, jobject objptr, jlong ptr, jint sid) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);

  model->Decode(sid);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpaonnx_OnlineRecognizer_isEndpoint(JNIEnv *env,
                                                      jobject objptr, jlong ptr,
                                                      jint sid) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);

  return model->IsEndpoint(sid);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL Java_com_k2fsa_sherpaonnx_OnlineRecognizer_isReady(
    JNIEnv *env, jobject objptr, jlong ptr, jint sid) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);

  return model->IsReady(sid);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpaonnx_OnlineRecognizer_newRecognizer(JNIEnv *env,
                                                         jobject objptr,
                                                         jobject _config) {
  auto config = sherpa_onnx::GetConfigJava(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::SherpaOnnx(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpaonnx_OnlineRecognizer_decodeStreams(JNIEnv *env,
                                                         jobject objptr,
                                                         jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  model->DecodeStreams();
  // TODO(zhaoming): for decode in parallel
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpaonnx_OnlineRecognizer_creatStream(
    JNIEnv *env, jobject objptr, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  int32_t sid = model->CreatStream();

  return sid;
}
SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpaonnx_OnlineRecognizer_releaseStreams(JNIEnv *env,
                                                          jobject objptr,
                                                          jlong ptr, jint sid) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);

  model->ReleaseStreams(sid);
}

// ********************************************************************

// see
// https://stackoverflow.com/questions/29043872/android-jni-return-multiple-variables
static jobject NewInteger(JNIEnv *env, int32_t value) {
  jclass cls = env->FindClass("java/lang/Integer");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(I)V");
  return env->NewObject(cls, constructor, value);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_WaveReader_00024Companion_readWave(
    JNIEnv *env, jclass /*cls*/, jobject asset_manager, jstring filename) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    exit(-1);
  }

  std::vector<char> buffer = sherpa_onnx::ReadFile(mgr, p_filename);

  std::istrstream is(buffer.data(), buffer.size());
#else
  std::ifstream is(p_filename, std::ios::binary);
#endif

  bool is_ok = false;
  int32_t sampling_rate = -1;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(is, &sampling_rate, &is_ok);

  env->ReleaseStringUTFChars(filename, p_filename);

  if (!is_ok) {
    SHERPA_ONNX_LOGE("Failed to read %s", p_filename);
    exit(-1);
  }

  jfloatArray ans = env->NewFloatArray(samples.size());
  env->SetFloatArrayRegion(ans, 0, samples.size(), samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, ans);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, sampling_rate));

  return obj_arr;
}

// wav reader for java interface
SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpaonnx_OnlineRecognizer_readWave(JNIEnv *env, jclass /*cls*/,
                                                    jstring filename) {
  return Java_com_k2fsa_sherpa_onnx_WaveReader_00024Companion_readWave(
      env, nullptr, nullptr, filename);
}