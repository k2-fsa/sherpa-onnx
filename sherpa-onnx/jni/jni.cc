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
#endif
#include <fstream>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/wave-reader.h"

#define SHERPA_ONNX_EXTERN_C extern "C"

namespace sherpa_onnx {

class SherpaOnnx {
 public:
#if __ANDROID_API__ >= 9
  SherpaOnnx(AAssetManager *mgr, const OnlineRecognizerConfig &config)
      : recognizer_(mgr, config), stream_(recognizer_.CreateStream()) {}
#endif

  explicit SherpaOnnx(const OnlineRecognizerConfig &config)
      : recognizer_(config), stream_(recognizer_.CreateStream()) {}

  void AcceptWaveform(int32_t sample_rate, const float *samples, int32_t n) {
    if (input_sample_rate_ == -1) {
      input_sample_rate_ = sample_rate;
    }

    stream_->AcceptWaveform(sample_rate, samples, n);
  }

  void InputFinished() const {
    std::vector<float> tail_padding(input_sample_rate_ * 0.6, 0);
    stream_->AcceptWaveform(input_sample_rate_, tail_padding.data(),
                            tail_padding.size());
    stream_->InputFinished();
  }

  const std::string GetText() const {
    auto result = recognizer_.GetResult(stream_.get());
    return result.text;
  }

  const std::vector<std::string> GetTokens() const {
    auto result = recognizer_.GetResult(stream_.get());
    return result.tokens;
  }

  bool IsEndpoint() const { return recognizer_.IsEndpoint(stream_.get()); }

  bool IsReady() const { return recognizer_.IsReady(stream_.get()); }

  void Reset() const { return recognizer_.Reset(stream_.get()); }

  void Decode() const { recognizer_.DecodeStream(stream_.get()); }

 private:
  OnlineRecognizer recognizer_;
  std::unique_ptr<OnlineStream> stream_;
  int32_t input_sample_rate_ = -1;
};

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
                        "Lcom/k2fsa/sherpa/onnx/OnlineModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  // transducer
  fid = env->GetFieldID(model_config_cls, "transducer",
                        "Lcom/k2fsa/sherpa/onnx/OnlineTransducerModelConfig;");
  jobject transducer_config = env->GetObjectField(model_config, fid);
  jclass transducer_config_cls = env->GetObjectClass(transducer_config);

  fid = env->GetFieldID(transducer_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(transducer_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(transducer_config_cls, "joiner", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.joiner = p;
  env->ReleaseStringUTFChars(s, p);

  // paraformer
  fid = env->GetFieldID(model_config_cls, "paraformer",
                        "Lcom/k2fsa/sherpa/onnx/OnlineParaformerModelConfig;");
  jobject paraformer_config = env->GetObjectField(model_config, fid);
  jclass paraformer_config_config_cls = env->GetObjectClass(paraformer_config);

  fid = env->GetFieldID(paraformer_config_config_cls, "encoder",
                        "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(paraformer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.paraformer.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(paraformer_config_config_cls, "decoder",
                        "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(paraformer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.paraformer.decoder = p;
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

  fid = env->GetFieldID(model_config_cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.provider = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "modelType", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.model_type = p;
  env->ReleaseStringUTFChars(s, p);

  //---------- rnn lm model config ----------
  fid = env->GetFieldID(cls, "lmConfig",
                        "Lcom/k2fsa/sherpa/onnx/OnlineLMConfig;");
  jobject lm_model_config = env->GetObjectField(config, fid);
  jclass lm_model_config_cls = env->GetObjectClass(lm_model_config);

  fid = env->GetFieldID(lm_model_config_cls, "model", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(lm_model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.lm_config.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(lm_model_config_cls, "scale", "F");
  ans.lm_config.scale = env->GetFloatField(lm_model_config, fid);

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
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_onnx::GetConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::SherpaOnnx(config);

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
  model->Reset();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_isReady(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  return model->IsReady();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_isEndpoint(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  return model->IsEndpoint();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_decode(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  model->Decode();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  model->AcceptWaveform(sample_rate, p, n);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_inputFinished(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr)->InputFinished();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_getText(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  // see
  // https://stackoverflow.com/questions/11621449/send-c-string-to-java-via-jni
  auto text = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr)->GetText();
  return env->NewStringUTF(text.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_getTokens(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto tokens = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr)->GetTokens();
  int size = tokens.size();
  jclass stringClass = env->FindClass("java/lang/String");

  // convert C++ list into jni string array
  jobjectArray result = env->NewObjectArray(size, stringClass, NULL);
  for (int i = 0; i < size; i++) {
    // Convert the C++ string to a C string
    const char *cstr = tokens[i].c_str();

    // Convert the C string to a jstring
    jstring jstr = env->NewStringUTF(cstr);

    // Set the array element
    env->SetObjectArrayElement(result, i, jstr);
  }

  return result;
}

// see
// https://stackoverflow.com/questions/29043872/android-jni-return-multiple-variables
static jobject NewInteger(JNIEnv *env, int32_t value) {
  jclass cls = env->FindClass("java/lang/Integer");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(I)V");
  return env->NewObject(cls, constructor, value);
}

static jobjectArray ReadWaveImpl(JNIEnv *env, std::istream &is,
                                 const char *p_filename) {
  bool is_ok = false;
  int32_t sampling_rate = -1;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(is, &sampling_rate, &is_ok);

  if (!is_ok) {
    SHERPA_ONNX_LOGE("Failed to read %s", p_filename);
    exit(-1);
  }

  jfloatArray samples_arr = env->NewFloatArray(samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, samples.size(), samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, samples_arr);
  env->SetObjectArrayElement(obj_arr, 1, NewInteger(env, sampling_rate));

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_WaveReader_00024Companion_readWaveFromFile(
    JNIEnv *env, jclass /*cls*/, jstring filename) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);
  std::ifstream is(p_filename, std::ios::binary);

  auto obj_arr = ReadWaveImpl(env, is, p_filename);

  env->ReleaseStringUTFChars(filename, p_filename);

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_WaveReader_00024Companion_readWaveFromAsset(
    JNIEnv *env, jclass /*cls*/, jobject asset_manager, jstring filename) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
    exit(-1);
  }
  SHERPA_ONNX_LOGE("Failed to read %s", p_filename);
  std::vector<char> buffer = sherpa_onnx::ReadFile(mgr, p_filename);

  std::istrstream is(buffer.data(), buffer.size());
#else
  std::ifstream is(p_filename, std::ios::binary);
#endif

  auto obj_arr = ReadWaveImpl(env, is, p_filename);

  env->ReleaseStringUTFChars(filename, p_filename);

  return obj_arr;
}

// ******warpper for OnlineRecognizer*******

// wav reader for java interface
SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_readWave(JNIEnv *env,
                                                     jclass /*cls*/,
                                                     jstring filename) {
  auto data =
      Java_com_k2fsa_sherpa_onnx_WaveReader_00024Companion_readWaveFromAsset(
          env, nullptr, nullptr, filename);
  return data;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL

Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_createOnlineRecognizer(

    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
  }
#endif
  sherpa_onnx::OnlineRecognizerConfig config =
      sherpa_onnx::GetConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto p_recognizer = new sherpa_onnx::OnlineRecognizer(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);
  return (jlong)p_recognizer;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_deleteOnlineRecognizer(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_createStream(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jlong ptr) {
  std::unique_ptr<sherpa_onnx::OnlineStream> s =
      reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr)->CreateStream();
  sherpa_onnx::OnlineStream *p_stream = s.release();
  return reinterpret_cast<jlong>(p_stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_isReady(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong s_ptr) {
  sherpa_onnx::OnlineRecognizer *model =
      reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  return model->IsReady(s);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_decodeStream(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong s_ptr) {
  sherpa_onnx::OnlineRecognizer *model =
      reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  model->DecodeStream(s);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_decodeStreams(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jlongArray ss_ptr,
                                                          jint stream_size) {
  sherpa_onnx::OnlineRecognizer *model =
      reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  jlong *p = env->GetLongArrayElements(ss_ptr, nullptr);
  jsize n = env->GetArrayLength(ss_ptr);
  std::vector<sherpa_onnx::OnlineStream *> p_ss(n);
  for (int32_t i = 0; i != n; ++i) {
    p_ss[i] = reinterpret_cast<sherpa_onnx::OnlineStream *>(p[i]);
  }

  model->DecodeStreams(p_ss.data(), n);
  env->ReleaseLongArrayElements(ss_ptr, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_getResult(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong s_ptr) {
  sherpa_onnx::OnlineRecognizer *model =
      reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  sherpa_onnx::OnlineRecognizerResult result = model->GetResult(s);
  return env->NewStringUTF(result.text.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_isEndpoint(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong s_ptr) {
  sherpa_onnx::OnlineRecognizer *model =
      reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  return model->IsEndpoint(s);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineRecognizer_reSet(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong s_ptr) {
  sherpa_onnx::OnlineRecognizer *model =
      reinterpret_cast<sherpa_onnx::OnlineRecognizer *>(ptr);
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  model->Reset(s);
}

// *********for OnlineStream *********
SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineStream_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong s_ptr, jint sample_rate,
    jfloatArray waveform) {
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  jfloat *p = env->GetFloatArrayElements(waveform, nullptr);
  jsize n = env->GetArrayLength(waveform);
  s->AcceptWaveform(sample_rate, p, n);
  env->ReleaseFloatArrayElements(waveform, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineStream_inputFinished(
    JNIEnv *env, jobject /*obj*/, jlong s_ptr) {
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  s->InputFinished();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineStream_deleteStream(
    JNIEnv *env, jobject /*obj*/, jlong s_ptr) {
  delete reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_onnx_OnlineStream_numFramesReady(
    JNIEnv *env, jobject /*obj*/, jlong s_ptr) {
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  return s->NumFramesReady();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_OnlineStream_isLastFrame(
    JNIEnv *env, jobject /*obj*/, jlong s_ptr, jint frame) {
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  return s->IsLastFrame(frame);
}
SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OnlineStream_reSet(
    JNIEnv *env, jobject /*obj*/, jlong s_ptr) {
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  s->Reset();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_onnx_OnlineStream_featureDim(
    JNIEnv *env, jobject /*obj*/, jlong s_ptr) {
  sherpa_onnx::OnlineStream *s =
      reinterpret_cast<sherpa_onnx::OnlineStream *>(s_ptr);
  return s->FeatureDim();
}
