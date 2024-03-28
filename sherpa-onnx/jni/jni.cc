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

#include <fstream>
#include <functional>
#include <strstream>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"
#include "sherpa-onnx/csrc/speaker-embedding-manager.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

#if SHERPA_ONNX_ENABLE_TTS == 1
#include "sherpa-onnx/csrc/offline-tts.h"
#endif

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

  std::string GetText() const {
    auto result = recognizer_.GetResult(stream_.get());
    return result.text;
  }

  const std::vector<std::string> GetTokens() const {
    auto result = recognizer_.GetResult(stream_.get());
    return result.tokens;
  }

  bool IsEndpoint() const { return recognizer_.IsEndpoint(stream_.get()); }

  bool IsReady() const { return recognizer_.IsReady(stream_.get()); }

  // If keywords is an empty string, it just recreates the decoding stream
  // If keywords is not empty, it will create a new decoding stream with
  // the given keywords appended to the default keywords.
  void Reset(bool recreate, const std::string &keywords = {}) {
    if (keywords.empty()) {
      if (recreate) {
        stream_ = recognizer_.CreateStream();
      } else {
        recognizer_.Reset(stream_.get());
      }
    } else {
      auto stream = recognizer_.CreateStream(keywords);
      // Set new keywords failed, the stream_ will not be updated.
      if (stream != nullptr) {
        stream_ = std::move(stream);
      } else {
        SHERPA_ONNX_LOGE("Failed to set keywords: %s", keywords.c_str());
      }
    }
  }

  void Decode() const { recognizer_.DecodeStream(stream_.get()); }

 private:
  OnlineRecognizer recognizer_;
  std::unique_ptr<OnlineStream> stream_;
  int32_t input_sample_rate_ = -1;
};

class SherpaOnnxOffline {
 public:
#if __ANDROID_API__ >= 9
  SherpaOnnxOffline(AAssetManager *mgr, const OfflineRecognizerConfig &config)
      : recognizer_(mgr, config) {}
#endif

  explicit SherpaOnnxOffline(const OfflineRecognizerConfig &config)
      : recognizer_(config) {}

  std::string Decode(int32_t sample_rate, const float *samples, int32_t n) {
    auto stream = recognizer_.CreateStream();
    stream->AcceptWaveform(sample_rate, samples, n);

    recognizer_.DecodeStream(stream.get());
    return stream->GetResult().text;
  }

 private:
  OfflineRecognizer recognizer_;
};

class SherpaOnnxVad {
 public:
#if __ANDROID_API__ >= 9
  SherpaOnnxVad(AAssetManager *mgr, const VadModelConfig &config)
      : vad_(mgr, config) {}
#endif

  explicit SherpaOnnxVad(const VadModelConfig &config) : vad_(config) {}

  void AcceptWaveform(const float *samples, int32_t n) {
    vad_.AcceptWaveform(samples, n);
  }

  bool Empty() const { return vad_.Empty(); }

  void Pop() { vad_.Pop(); }

  void Clear() { vad_.Clear(); }

  const SpeechSegment &Front() const { return vad_.Front(); }

  bool IsSpeechDetected() const { return vad_.IsSpeechDetected(); }

  void Reset() { vad_.Reset(); }

 private:
  VoiceActivityDetector vad_;
};

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

class SherpaOnnxSpeakerEmbeddingExtractorStream {
 public:
  explicit SherpaOnnxSpeakerEmbeddingExtractorStream(
      std::unique_ptr<OnlineStream> stream)
      : stream_(std::move(stream)) {}

  void AcceptWaveform(int32_t sample_rate, const float *samples,
                      int32_t n) const {
    stream_->AcceptWaveform(sample_rate, samples, n);
  }

  void InputFinished() const { stream_->InputFinished(); }

  OnlineStream *Get() const { return stream_.get(); }

 private:
  std::unique_ptr<OnlineStream> stream_;
};

class SherpaOnnxSpeakerEmbeddingExtractor {
 public:
#if __ANDROID_API__ >= 9
  SherpaOnnxSpeakerEmbeddingExtractor(
      AAssetManager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : extractor_(mgr, config) {}
#endif

  explicit SherpaOnnxSpeakerEmbeddingExtractor(
      const SpeakerEmbeddingExtractorConfig &config)
      : extractor_(config) {}

  int32_t Dim() const { return extractor_.Dim(); }

  bool IsReady(const SherpaOnnxSpeakerEmbeddingExtractorStream *stream) const {
    return extractor_.IsReady(stream->Get());
  }

  SherpaOnnxSpeakerEmbeddingExtractorStream *CreateStream() const {
    return new SherpaOnnxSpeakerEmbeddingExtractorStream(
        extractor_.CreateStream());
  }

  std::vector<float> Compute(
      const SherpaOnnxSpeakerEmbeddingExtractorStream *stream) const {
    return extractor_.Compute(stream->Get());
  }

 private:
  SpeakerEmbeddingExtractor extractor_;
};

static SpeakerEmbeddingExtractorConfig GetSpeakerEmbeddingExtractorConfig(
    JNIEnv *env, jobject config) {
  SpeakerEmbeddingExtractorConfig ans;

  jclass cls = env->GetObjectClass(config);

  jfieldID fid = env->GetFieldID(cls, "model", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);

  ans.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "numThreads", "I");
  ans.num_threads = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "debug", "Z");
  ans.debug = env->GetBooleanField(config, fid);

  fid = env->GetFieldID(cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.provider = p;
  env->ReleaseStringUTFChars(s, p);

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

  fid = env->GetFieldID(cls, "hotwordsFile", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.hotwords_file = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "hotwordsScore", "F");
  ans.hotwords_score = env->GetFloatField(config, fid);

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
  jclass paraformer_config_cls = env->GetObjectClass(paraformer_config);

  fid = env->GetFieldID(paraformer_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(paraformer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.paraformer.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(paraformer_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(paraformer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.paraformer.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  // streaming zipformer2 CTC
  fid =
      env->GetFieldID(model_config_cls, "zipformer2Ctc",
                      "Lcom/k2fsa/sherpa/onnx/OnlineZipformer2CtcModelConfig;");
  jobject zipformer2_ctc_config = env->GetObjectField(model_config, fid);
  jclass zipformer2_ctc_config_cls = env->GetObjectClass(zipformer2_ctc_config);

  fid =
      env->GetFieldID(zipformer2_ctc_config_cls, "model", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(zipformer2_ctc_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.zipformer2_ctc.model = p;
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

static OfflineRecognizerConfig GetOfflineConfig(JNIEnv *env, jobject config) {
  OfflineRecognizerConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  //---------- decoding ----------
  fid = env->GetFieldID(cls, "decodingMethod", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.decoding_method = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "maxActivePaths", "I");
  ans.max_active_paths = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "hotwordsFile", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.hotwords_file = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "hotwordsScore", "F");
  ans.hotwords_score = env->GetFloatField(config, fid);

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
                        "Lcom/k2fsa/sherpa/onnx/OfflineModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

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

  // transducer
  fid = env->GetFieldID(model_config_cls, "transducer",
                        "Lcom/k2fsa/sherpa/onnx/OfflineTransducerModelConfig;");
  jobject transducer_config = env->GetObjectField(model_config, fid);
  jclass transducer_config_cls = env->GetObjectClass(transducer_config);

  fid = env->GetFieldID(transducer_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.encoder_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(transducer_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.decoder_filename = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(transducer_config_cls, "joiner", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(transducer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.transducer.joiner_filename = p;
  env->ReleaseStringUTFChars(s, p);

  // paraformer
  fid = env->GetFieldID(model_config_cls, "paraformer",
                        "Lcom/k2fsa/sherpa/onnx/OfflineParaformerModelConfig;");
  jobject paraformer_config = env->GetObjectField(model_config, fid);
  jclass paraformer_config_cls = env->GetObjectClass(paraformer_config);

  fid = env->GetFieldID(paraformer_config_cls, "model", "Ljava/lang/String;");

  s = (jstring)env->GetObjectField(paraformer_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.paraformer.model = p;
  env->ReleaseStringUTFChars(s, p);

  // whisper
  fid = env->GetFieldID(model_config_cls, "whisper",
                        "Lcom/k2fsa/sherpa/onnx/OfflineWhisperModelConfig;");
  jobject whisper_config = env->GetObjectField(model_config, fid);
  jclass whisper_config_cls = env->GetObjectClass(whisper_config);

  fid = env->GetFieldID(whisper_config_cls, "encoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.whisper.encoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_config_cls, "decoder", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.whisper.decoder = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_config_cls, "language", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.whisper.language = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_config_cls, "task", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(whisper_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.whisper.task = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(whisper_config_cls, "tailPaddings", "I");
  ans.model_config.whisper.tail_paddings =
      env->GetIntField(whisper_config, fid);

  return ans;
}

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

static VadModelConfig GetVadModelConfig(JNIEnv *env, jobject config) {
  VadModelConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  // silero_vad
  fid = env->GetFieldID(cls, "sileroVadModelConfig",
                        "Lcom/k2fsa/sherpa/onnx/SileroVadModelConfig;");
  jobject silero_vad_config = env->GetObjectField(config, fid);
  jclass silero_vad_config_cls = env->GetObjectClass(silero_vad_config);

  fid = env->GetFieldID(silero_vad_config_cls, "model", "Ljava/lang/String;");
  auto s = (jstring)env->GetObjectField(silero_vad_config, fid);
  auto p = env->GetStringUTFChars(s, nullptr);
  ans.silero_vad.model = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(silero_vad_config_cls, "threshold", "F");
  ans.silero_vad.threshold = env->GetFloatField(silero_vad_config, fid);

  fid = env->GetFieldID(silero_vad_config_cls, "minSilenceDuration", "F");
  ans.silero_vad.min_silence_duration =
      env->GetFloatField(silero_vad_config, fid);

  fid = env->GetFieldID(silero_vad_config_cls, "minSpeechDuration", "F");
  ans.silero_vad.min_speech_duration =
      env->GetFloatField(silero_vad_config, fid);

  fid = env->GetFieldID(silero_vad_config_cls, "windowSize", "I");
  ans.silero_vad.window_size = env->GetIntField(silero_vad_config, fid);

  fid = env->GetFieldID(cls, "sampleRate", "I");
  ans.sample_rate = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "numThreads", "I");
  ans.num_threads = env->GetIntField(config, fid);

  fid = env->GetFieldID(cls, "provider", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.provider = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "debug", "Z");
  ans.debug = env->GetBooleanField(config, fid);

  return ans;
}

#if SHERPA_ONNX_ENABLE_TTS == 1
class SherpaOnnxOfflineTts {
 public:
#if __ANDROID_API__ >= 9
  SherpaOnnxOfflineTts(AAssetManager *mgr, const OfflineTtsConfig &config)
      : tts_(mgr, config) {}
#endif
  explicit SherpaOnnxOfflineTts(const OfflineTtsConfig &config)
      : tts_(config) {}

  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      std::function<void(const float *, int32_t, float)> callback = nullptr) const {
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

  fid = env->GetFieldID(cls, "maxNumSentences", "I");
  ans.max_num_sentences = env->GetIntField(config, fid);

  return ans;
}
#endif

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractor_new(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jobject asset_manager,
                                                         jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
  }
#endif
  auto config = sherpa_onnx::GetSpeakerEmbeddingExtractorConfig(env, _config);
  SHERPA_ONNX_LOGE("new config:\n%s", config.ToString().c_str());

  auto extractor = new sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractor(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)extractor;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractor_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_onnx::GetSpeakerEmbeddingExtractorConfig(env, _config);
  SHERPA_ONNX_LOGE("newFromFile config:\n%s", config.ToString().c_str());

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Errors found in config!");
  }

  auto extractor = new sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractor(config);

  return (jlong)extractor;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractor_delete(JNIEnv *env,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractor *>(
      ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractor_createStream(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto stream =
      reinterpret_cast<sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractor *>(ptr)
          ->CreateStream();

  return (jlong)stream;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractor_isReady(JNIEnv *env,
                                                             jobject /*obj*/,
                                                             jlong ptr,
                                                             jlong stream_ptr) {
  auto extractor =
      reinterpret_cast<sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractor *>(ptr);
  auto stream = reinterpret_cast<
      sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractorStream *>(stream_ptr);
  return extractor->IsReady(stream);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jfloatArray JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractor_compute(JNIEnv *env,
                                                             jobject /*obj*/,
                                                             jlong ptr,
                                                             jlong stream_ptr) {
  auto extractor =
      reinterpret_cast<sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractor *>(ptr);
  auto stream = reinterpret_cast<
      sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractorStream *>(stream_ptr);

  std::vector<float> embedding = extractor->Compute(stream);
  jfloatArray embedding_arr = env->NewFloatArray(embedding.size());
  env->SetFloatArrayRegion(embedding_arr, 0, embedding.size(),
                           embedding.data());
  return embedding_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractor_dim(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto extractor =
      reinterpret_cast<sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractor *>(ptr);
  return extractor->Dim();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractorStream_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<
      sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractorStream *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractorStream_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto stream = reinterpret_cast<
      sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractorStream *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  stream->AcceptWaveform(sample_rate, p, n);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingExtractorStream_inputFinished(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto stream = reinterpret_cast<
      sherpa_onnx::SherpaOnnxSpeakerEmbeddingExtractorStream *>(ptr);
  stream->InputFinished();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_new(
    JNIEnv *env, jobject /*obj*/, jint dim) {
  auto p = new sherpa_onnx::SpeakerEmbeddingManager(dim);
  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_delete(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);
  delete manager;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_add(JNIEnv *env,
                                                       jobject /*obj*/,
                                                       jlong ptr, jstring name,
                                                       jfloatArray embedding) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize n = env->GetArrayLength(embedding);

  if (n != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(n));
    exit(-1);
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Add(p_name, p);
  env->ReleaseStringUTFChars(name, p_name);
  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_addList(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring name,
    jobjectArray embedding_arr) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  int num_embeddings = env->GetArrayLength(embedding_arr);
  if (num_embeddings == 0) {
    return false;
  }

  std::vector<std::vector<float>> embedding_list;
  embedding_list.reserve(num_embeddings);
  for (int32_t i = 0; i != num_embeddings; ++i) {
    jfloatArray embedding =
        (jfloatArray)env->GetObjectArrayElement(embedding_arr, i);

    jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
    jsize n = env->GetArrayLength(embedding);

    if (n != manager->Dim()) {
      SHERPA_ONNX_LOGE("i: %d. Expected dim %d, given %d", i, manager->Dim(),
                       static_cast<int32_t>(n));
      exit(-1);
    }

    embedding_list.push_back({p, p + n});
    env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Add(p_name, embedding_list);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_remove(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jstring name) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Remove(p_name);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_search(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jfloatArray embedding,
                                                          jfloat threshold) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize n = env->GetArrayLength(embedding);

  if (n != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(n));
    exit(-1);
  }

  std::string name = manager->Search(p, threshold);

  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  return env->NewStringUTF(name.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_verify(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring name,
    jfloatArray embedding, jfloat threshold) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize n = env->GetArrayLength(embedding);

  if (n != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(n));
    exit(-1);
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Verify(p_name, p, threshold);

  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_contains(JNIEnv *env,
                                                            jobject /*obj*/,
                                                            jlong ptr,
                                                            jstring name) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Contains(p_name);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_numSpeakers(JNIEnv *env,
                                                               jobject /*obj*/,
                                                               jlong ptr) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);
  return manager->NumSpeakers();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_allSpeakerNames(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);
  std::vector<std::string> all_speakers = manager->GetAllSpeakers();

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      all_speakers.size(), env->FindClass("java/lang/String"), nullptr);

  int32_t i = 0;
  for (auto &s : all_speakers) {
    jstring js = env->NewStringUTF(s.c_str());
    env->SetObjectArrayElement(obj_arr, i, js);

    ++i;
  }

  return obj_arr;
}

// see
// https://stackoverflow.com/questions/29043872/android-jni-return-multiple-variables
static jobject NewInteger(JNIEnv *env, int32_t value) {
  jclass cls = env->FindClass("java/lang/Integer");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(I)V");
  return env->NewObject(cls, constructor, value);
}

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
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_Vad_new(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
  }
#endif
  auto config = sherpa_onnx::GetVadModelConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::SherpaOnnxVad(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_Vad_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  auto config = sherpa_onnx::GetVadModelConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::SherpaOnnxVad(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_Vad_delete(JNIEnv *env,
                                                             jobject /*obj*/,
                                                             jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::SherpaOnnxVad *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_Vad_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxVad *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  model->AcceptWaveform(p, n);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_Vad_empty(JNIEnv *env,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxVad *>(ptr);
  return model->Empty();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_Vad_pop(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxVad *>(ptr);
  model->Pop();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_Vad_clear(JNIEnv *env,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxVad *>(ptr);
  model->Clear();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_Vad_front(JNIEnv *env, jobject /*obj*/, jlong ptr) {
  const auto &front =
      reinterpret_cast<sherpa_onnx::SherpaOnnxVad *>(ptr)->Front();

  jfloatArray samples_arr = env->NewFloatArray(front.samples.size());
  env->SetFloatArrayRegion(samples_arr, 0, front.samples.size(),
                           front.samples.data());

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      2, env->FindClass("java/lang/Object"), nullptr);

  env->SetObjectArrayElement(obj_arr, 0, NewInteger(env, front.start));
  env->SetObjectArrayElement(obj_arr, 1, samples_arr);

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_onnx_Vad_isSpeechDetected(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxVad *>(ptr);
  return model->IsSpeechDetected();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_Vad_reset(JNIEnv *env,
                                                            jobject /*obj*/,
                                                            jlong ptr) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxVad *>(ptr);
  model->Reset();
}

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
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxOffline_new(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_ONNX_LOGE("Failed to get asset manager: %p", mgr);
  }
#endif
  auto config = sherpa_onnx::GetOfflineConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::SherpaOnnxOffline(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_SherpaOnnxOffline_newFromFile(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jobject _config) {
  auto config = sherpa_onnx::GetOfflineConfig(env, _config);
  SHERPA_ONNX_LOGE("config:\n%s", config.ToString().c_str());
  auto model = new sherpa_onnx::SherpaOnnxOffline(config);

  return (jlong)model;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxOffline_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::SherpaOnnxOffline *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnx_reset(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jboolean recreate,
    jstring keywords) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnx *>(ptr);
  const char *p_keywords = env->GetStringUTFChars(keywords, nullptr);
  model->Reset(recreate, p_keywords);
  env->ReleaseStringUTFChars(keywords, p_keywords);
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
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpa_onnx_SherpaOnnxOffline_decode(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto model = reinterpret_cast<sherpa_onnx::SherpaOnnxOffline *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  auto text = model->Decode(sample_rate, p, n);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);

  return env->NewStringUTF(text.c_str());
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
