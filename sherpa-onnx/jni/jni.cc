// sherpa-onnx/jni/jni.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation
//                2022       Pingfeng Luo
//                2023       Zhaoming

// TODO(fangjun): Add documentation to functions/methods in this file
// and also show how to use them with kotlin, possibly with java.

#include <fstream>
#include <functional>
#include <strstream>
#include <utility>

#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

class SherpaOnnxKws {
 public:
#if __ANDROID_API__ >= 9
  SherpaOnnxKws(AAssetManager *mgr, const KeywordSpotterConfig &config)
      : keyword_spotter_(mgr, config),
        stream_(keyword_spotter_.CreateStream()) {}
#endif

  explicit SherpaOnnxKws(const KeywordSpotterConfig &config)
      : keyword_spotter_(config), stream_(keyword_spotter_.CreateStream()) {}

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

  // If keywords is an empty string, it just recreates the decoding stream
  // always returns true in this case.
  // If keywords is not empty, it will create a new decoding stream with
  // the given keywords appended to the default keywords.
  // Return false if errors occurred when adding keywords, true otherwise.
  bool Reset(const std::string &keywords = {}) {
    if (keywords.empty()) {
      stream_ = keyword_spotter_.CreateStream();
      return true;
    } else {
      auto stream = keyword_spotter_.CreateStream(keywords);
      // Set new keywords failed, the stream_ will not be updated.
      if (stream == nullptr) {
        return false;
      } else {
        stream_ = std::move(stream);
        return true;
      }
    }
  }

  std::string GetKeyword() const {
    auto result = keyword_spotter_.GetResult(stream_.get());
    return result.keyword;
  }

  std::vector<std::string> GetTokens() const {
    auto result = keyword_spotter_.GetResult(stream_.get());
    return result.tokens;
  }

  bool IsReady() const { return keyword_spotter_.IsReady(stream_.get()); }

  void Decode() const { keyword_spotter_.DecodeStream(stream_.get()); }

 private:
  KeywordSpotter keyword_spotter_;
  std::unique_ptr<OnlineStream> stream_;
  int32_t input_sample_rate_ = -1;
};

static KeywordSpotterConfig GetKwsConfig(JNIEnv *env, jobject config) {
  KeywordSpotterConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  // https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html
  // https://courses.cs.washington.edu/courses/cse341/99wi/java/tutorial/native1.1/implementing/field.html

  //---------- decoding ----------
  fid = env->GetFieldID(cls, "maxActivePaths", "I");
  ans.max_active_paths = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "keywordsFile", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.keywords_file = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "keywordsScore", "F");
  ans.keywords_score = env->GetFloatField(config, fid);

  fid = env->GetFieldID(cls, "keywordsThreshold", "F");
  ans.keywords_threshold = env->GetFloatField(config, fid);

  fid = env->GetFieldID(cls, "numTrailingBlanks", "I");
  ans.num_trailing_blanks = env->GetIntField(config, fid);

  //---------- feat config ----------
  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpa/onnx/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  fid = env->GetFieldID(feat_config_cls, "sampleRate", "I");
  ans.feat_config.sampling_rate = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "featureDim", "I");
  ans.feat_config.feature_dim = env->GetIntField(feat_config, fid);

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

  return ans;
}

}  // namespace sherpa_onnx

// see
// https://stackoverflow.com/questions/29043872/android-jni-return-multiple-variables
jobject NewInteger(JNIEnv *env, int32_t value) {
  jclass cls = env->FindClass("java/lang/Integer");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(I)V");
  return env->NewObject(cls, constructor, value);
}

jobject NewFloat(JNIEnv *env, float value) {
  jclass cls = env->FindClass("java/lang/Float");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(F)V");
  return env->NewObject(cls, constructor, value);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL Java_com_k2fsa_sherpa_onnx_GeneratedAudio_saveImpl(
    JNIEnv *env, jobject /*obj*/, jstring filename, jfloatArray samples,
    jint sample_rate) {
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  bool ok = sherpa_onnx::WriteWave(p_filename, sample_rate, p, n);

  env->ReleaseStringUTFChars(filename, p_filename);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_new(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
  }
#endif
  auto config = sherpa_onnx::GetKwsConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::SherpaOnnxKws(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_onnx::GetKwsConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::SherpaOnnxKws(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::SherpaOnnxKws *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_isReady(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxKws *>(ptr);
  return model->IsReady();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_decode(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxKws *>(ptr);
  model->Decode();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxKws *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  model->AcceptWaveform(sample_rate, p, n);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_inputFinished(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  reinterpret_cast<sherpa_onnx::SherpaOnnxKws *>(ptr)->InputFinished();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_getKeyword(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  // see
  // https://stackoverflow.com/questions/11621449/send-c-string-to-java-via-jni
  auto text = reinterpret_cast<sherpa_onnx::SherpaOnnxKws *>(ptr)->GetKeyword();
  return env->NewStringUTF(text.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_reset(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring keywords) {
  const char *p_keywords = env->GetStringUTFChars(keywords, nullptr);

  std::string keywords_str = p_keywords;

  bool status =
      reinterpret_cast<sherpa_onnx::SherpaOnnxKws *>(ptr)->Reset(keywords_str);
  env->ReleaseStringUTFChars(keywords, p_keywords);
  return status;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_SherpaOnnxKws_getTokens(JNIEnv *env, jobject /*obj*/,
                                                   jlong ptr) {
  auto tokens =
      reinterpret_cast<sherpa_onnx::SherpaOnnxKws *>(ptr)->GetTokens();
  int32_t size = tokens.size();
  jclass stringClass = env->FindClass("java/lang/String");

  // convert C++ list into jni string array
  jobjectArray result = env->NewObjectArray(size, stringClass, nullptr);
  for (int32_t i = 0; i < size; i++) {
    // Convert the C++ string to a C string
    const char *cstr = tokens[i].c_str();

    // Convert the C string to a jstring
    jstring jstr = env->NewStringUTF(cstr);

    // Set the array element
    env->SetObjectArrayElement(result, i, jstr);
  }

  return result;
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

#if 0
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
#endif
