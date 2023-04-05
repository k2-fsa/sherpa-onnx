// sherpa-onnx/jni/jni_java.cc
// It is based on jni.cc, but with some changes
// Copyright   2022-2023  by zhaoming
// this will be used for the java environment
// com.k2fsa.sherpaonnx.rcglib.OnlineRecognizer
// It supports parallel stream decoding and each
// stream will assign a uuid id.
//

#include "jni.h"  // NOLINT

#include <strstream>
#include <utility>

#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

#define SHERPA_ONNX_EXTERN_C extern "C"
#include <mutex>  // NOLINT
namespace sherpa_onnx {

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_int_distribution<> dis(0, 15);
static std::uniform_int_distribution<> dis2(8, 11);

std::string generate_uuid() {
  // generate uuid for streams
  std::stringstream ss;
  int i;
  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 4; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 12; i++) {
    ss << dis(gen);
  }
  return ss.str();
}

class JavaRecognizer {
  // the recognizer use std::map to manage streams based on its uuid,
  // it can be used for parallel stream decode
 public:
  JavaRecognizer(

      const sherpa_onnx::OnlineRecognizerConfig &config)
      : recognizer_(config) {}

  void AcceptWaveform(int32_t sample_rate, const float *samples, int32_t n,
                      std::string s_id) {
    if (input_sample_rate_ == -1) {
      input_sample_rate_ = sample_rate;
    }
    auto stream_ = findStream(s_id);
    stream_->AcceptWaveform(sample_rate, samples, n);
  }

  void InputFinished(std::string s_id) const {
    std::shared_ptr<OnlineStream> stream_ = findStream(s_id);
    std::vector<float> tail_padding(input_sample_rate_ * 0.32, 0);
    stream_->AcceptWaveform(input_sample_rate_, tail_padding.data(),
                            tail_padding.size());
    stream_->InputFinished();
  }

  const std::string GetText(std::string s_id) const {
    auto stream_ = findStream(s_id);
    auto result = recognizer_.GetResult(stream_.get());
    return result.text;
  }

  bool IsEndpoint(std::string s_id) const {
    auto stream_ = findStream(s_id);
    return recognizer_.IsEndpoint(stream_.get());
  }

  bool IsReady(std::string s_id) const {
    auto stream_ = findStream(s_id);
    return recognizer_.IsReady(stream_.get());
  }

  void Reset(std::string s_id) const {
    auto stream_ = findStream(s_id);
    return recognizer_.Reset(stream_.get());
  }

  void Decode(std::string s_id) const {
    auto stream_ = findStream(s_id);
    recognizer_.DecodeStream(stream_.get());
  }

  std::string CreatStream() {
    // create a new stream for recognizer
    std::unique_lock<std::mutex> lock(mutex_);
    std::shared_ptr<OnlineStream> s = recognizer_.CreateStream();
    std::string s_key = generate_uuid();
    s_map.insert({s_key, std::move(s)});

    return s_key;
  }
  void DecodeStreams() const {
    // not implemented
  }
  void ReleaseStreams(std::string s_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto iter = s_map.find(s_id);
    if (iter != s_map.end()) {
      s_map.erase(iter);
    }
  }

 private:
  std::shared_ptr<OnlineStream> findStream(std::string s_id) const {
    auto iter = s_map.find(s_id);

    if (iter != s_map.end())

      return iter->second;

    else
      SHERPA_ONNX_LOGE("Do not Find stream id\n%s", s_id.c_str());

    return nullptr;
  }
  sherpa_onnx::OnlineRecognizer recognizer_;
  std::map<std::string, std::shared_ptr<OnlineStream>> s_map;
  int32_t input_sample_rate_ = -1;
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
                        "Lcom/k2fsa/sherpaonnx/rcglib/FeatureConfig;");
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
                        "Lcom/k2fsa/sherpaonnx/rcglib/EndpointConfig;");
  jobject endpoint_config = env->GetObjectField(config, fid);
  jclass endpoint_config_cls = env->GetObjectClass(endpoint_config);

  fid = env->GetFieldID(endpoint_config_cls, "rule1",
                        "Lcom/k2fsa/sherpaonnx/rcglib/EndpointRule;");
  jobject rule1 = env->GetObjectField(endpoint_config, fid);
  jclass rule_class = env->GetObjectClass(rule1);

  fid = env->GetFieldID(endpoint_config_cls, "rule2",
                        "Lcom/k2fsa/sherpaonnx/rcglib/EndpointRule;");
  jobject rule2 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(endpoint_config_cls, "rule3",
                        "Lcom/k2fsa/sherpaonnx/rcglib/EndpointRule;");
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
  fid = env->GetFieldID(
      cls, "modelConfig",
      "Lcom/k2fsa/sherpaonnx/rcglib/OnlineTransducerModelConfig;");
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



// all inferface are used for JNI in java
SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_acceptWaveform(
    JNIEnv *env, jobject objptr, jlong ptr, jfloatArray samples,
    jint sample_rate, jstring sid) {
  auto model = reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr);

  // jstring cast to string need this operation
  const char *sp = env->GetStringUTFChars(sid, nullptr);
  std::string new_sid = std::string(sp);
  env->ReleaseStringUTFChars(sid, sp);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  model->AcceptWaveform(sample_rate, p, n, new_sid);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_inputFinished(
    JNIEnv *env, jobject objptr, jlong ptr, jstring sid) {
  const char *p = env->GetStringUTFChars(sid, nullptr);
  std::string new_sid = std::string(p);
  env->ReleaseStringUTFChars(sid, p);
  reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr)->InputFinished(new_sid);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_getText(JNIEnv *env,
                                                              jobject objptr,
                                                              jlong ptr,
                                                              jstring sid) {
  const char *p = env->GetStringUTFChars(sid, nullptr);
  std::string new_sid = std::string(p);
  env->ReleaseStringUTFChars(sid, p);
  auto text =
      reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr)->GetText(new_sid);
  return env->NewStringUTF(text.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_reset(JNIEnv *env,
                                                            jobject objptr,
                                                            jlong ptr,
                                                            jstring sid) {
  auto model = reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr);
  const char *p = env->GetStringUTFChars(sid, nullptr);
  std::string new_sid = std::string(p);
  env->ReleaseStringUTFChars(sid, p);
  model->Reset(new_sid);

  // SHERPA_ONNX_LOGE("reset sid:\n%s", new_sid.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_decode(JNIEnv *env,
                                                             jobject objptr,
                                                             jlong ptr,
                                                             jstring sid) {
  auto model = reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr);
  const char *p = env->GetStringUTFChars(sid, nullptr);
  std::string new_sid = std::string(p);
  env->ReleaseStringUTFChars(sid, p);
  model->Decode(new_sid);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_isEndpoint(JNIEnv *env,
                                                                 jobject objptr,
                                                                 jlong ptr,
                                                                 jstring sid) {
  auto model = reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr);
  const char *p = env->GetStringUTFChars(sid, nullptr);
  std::string new_sid = std::string(p);
  env->ReleaseStringUTFChars(sid, p);
  return model->IsEndpoint(new_sid);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_isReady(JNIEnv *env,
                                                              jobject objptr,
                                                              jlong ptr,
                                                              jstring sid) {
  auto model = reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr);
  const char *p = env->GetStringUTFChars(sid, nullptr);
  std::string new_sid = std::string(p);
  env->ReleaseStringUTFChars(sid, p);

  return model->IsReady(new_sid);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_newRecognizer(
    JNIEnv *env, jobject objptr, jobject _config) {
  auto config = sherpa_onnx::GetConfigJava(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::JavaRecognizer(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_decodeStreams(
    JNIEnv *env, jobject objptr, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr);
  model->DecodeStreams();
  // TODO(zhaoming): for decode in parallel
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_creatStream(
    JNIEnv *env, jobject objptr, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr);
  std::string sid = model->CreatStream();

  return env->NewStringUTF(sid.c_str());
}
SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpaonnx_rcglib_OnlineRecognizer_releaseStreams(
    JNIEnv *env, jobject objptr, jlong ptr, jstring sid) {
  auto model = reinterpret_cast<sherpa_onnx::JavaRecognizer *>(ptr);
  const char *p = env->GetStringUTFChars(sid, nullptr);
  std::string new_sid = std::string(p);
  env->ReleaseStringUTFChars(sid, p);

  model->ReleaseStreams(new_sid);
}
