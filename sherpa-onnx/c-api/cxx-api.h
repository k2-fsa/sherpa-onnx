// sherpa-onnx/c-api/cxx-api.h
//
// Copyright (c)  2024  Xiaomi Corporation

// C++ Wrapper of the C API for sherpa-onnx
#ifndef SHERPA_ONNX_C_API_CXX_API_H_
#define SHERPA_ONNX_C_API_CXX_API_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"

namespace sherpa_onnx::cxx {

// ============================================================================
// Streaming ASR
// ============================================================================
struct OnlineTransducerModelConfig {
  std::string encoder;
  std::string decoder;
  std::string joiner;
};

struct OnlineParaformerModelConfig {
  std::string encoder;
  std::string decoder;
};

struct OnlineZipformer2CtcModelConfig {
  std::string model;
};

struct OnlineNemoCtcModelConfig {
  std::string model;
};

struct OnlineToneCtcModelConfig {
  std::string model;
};

struct OnlineModelConfig {
  OnlineTransducerModelConfig transducer;
  OnlineParaformerModelConfig paraformer;
  OnlineZipformer2CtcModelConfig zipformer2_ctc;
  OnlineNemoCtcModelConfig nemo_ctc;
  OnlineToneCtcModelConfig t_one_ctc;
  std::string tokens;
  int32_t num_threads = 1;
  std::string provider = "cpu";
  bool debug = false;
  std::string model_type;
  std::string modeling_unit = "cjkchar";
  std::string bpe_vocab;
  std::string tokens_buf;
};

struct FeatureConfig {
  int32_t sample_rate = 16000;
  int32_t feature_dim = 80;
};

struct OnlineCtcFstDecoderConfig {
  std::string graph;
  int32_t max_active = 3000;
};

struct HomophoneReplacerConfig {
  std::string dict_dir;  // unused
  std::string lexicon;
  std::string rule_fsts;
};

struct OnlineRecognizerConfig {
  FeatureConfig feat_config;
  OnlineModelConfig model_config;

  std::string decoding_method = "greedy_search";

  int32_t max_active_paths = 4;

  bool enable_endpoint = false;

  float rule1_min_trailing_silence = 2.4;

  float rule2_min_trailing_silence = 1.2;

  float rule3_min_utterance_length = 20;

  std::string hotwords_file;

  float hotwords_score = 1.5;

  OnlineCtcFstDecoderConfig ctc_fst_decoder_config;
  std::string rule_fsts;
  std::string rule_fars;
  float blank_penalty = 0;

  std::string hotwords_buf;
  HomophoneReplacerConfig hr;
};

struct OnlineRecognizerResult {
  std::string text;
  std::vector<std::string> tokens;
  std::vector<float> timestamps;
  std::string json;
};

struct Wave {
  std::vector<float> samples;
  int32_t sample_rate;
};

SHERPA_ONNX_API Wave ReadWave(const std::string &filename);

// Return true on success;
// Return false on failure
SHERPA_ONNX_API bool WriteWave(const std::string &filename, const Wave &wave);

template <typename Derived, typename T>
class SHERPA_ONNX_API MoveOnly {
 public:
  MoveOnly() = default;
  explicit MoveOnly(const T *p) : p_(p) {}

  ~MoveOnly() { Destroy(); }

  MoveOnly(const MoveOnly &) = delete;

  MoveOnly &operator=(const MoveOnly &) = delete;

  MoveOnly(MoveOnly &&other) : p_(other.Release()) {}

  MoveOnly &operator=(MoveOnly &&other) {
    if (&other == this) {
      return *this;
    }

    Destroy();

    p_ = other.Release();

    return *this;
  }

  const T *Get() const { return p_; }

  const T *Release() {
    const T *p = p_;
    p_ = nullptr;
    return p;
  }

 private:
  void Destroy() {
    if (p_ == nullptr) {
      return;
    }

    static_cast<Derived *>(this)->Destroy(p_);

    p_ = nullptr;
  }

 protected:
  const T *p_ = nullptr;
};

class SHERPA_ONNX_API OnlineStream
    : public MoveOnly<OnlineStream, SherpaOnnxOnlineStream> {
 public:
  explicit OnlineStream(const SherpaOnnxOnlineStream *p);

  void AcceptWaveform(int32_t sample_rate, const float *samples,
                      int32_t n) const;

  void InputFinished() const;

  void Destroy(const SherpaOnnxOnlineStream *p) const;
};

class SHERPA_ONNX_API OnlineRecognizer
    : public MoveOnly<OnlineRecognizer, SherpaOnnxOnlineRecognizer> {
 public:
  static OnlineRecognizer Create(const OnlineRecognizerConfig &config);

  void Destroy(const SherpaOnnxOnlineRecognizer *p) const;

  OnlineStream CreateStream() const;

  OnlineStream CreateStream(const std::string &hotwords) const;

  bool IsReady(const OnlineStream *s) const;

  void Decode(const OnlineStream *s) const;

  void Decode(const OnlineStream *ss, int32_t n) const;

  OnlineRecognizerResult GetResult(const OnlineStream *s) const;

  void Reset(const OnlineStream *s) const;

  bool IsEndpoint(const OnlineStream *s) const;

 private:
  explicit OnlineRecognizer(const SherpaOnnxOnlineRecognizer *p);
};

// ============================================================================
// Non-streaming ASR
// ============================================================================
struct SHERPA_ONNX_API OfflineTransducerModelConfig {
  std::string encoder;
  std::string decoder;
  std::string joiner;
};

struct SHERPA_ONNX_API OfflineParaformerModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineNemoEncDecCtcModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineWhisperModelConfig {
  std::string encoder;
  std::string decoder;
  std::string language;
  std::string task = "transcribe";
  int32_t tail_paddings = -1;
};

struct SHERPA_ONNX_API OfflineCanaryModelConfig {
  std::string encoder;
  std::string decoder;
  std::string src_lang;
  std::string tgt_lang;
  bool use_pnc = true;
};

struct SHERPA_ONNX_API OfflineFireRedAsrModelConfig {
  std::string encoder;
  std::string decoder;
};

struct SHERPA_ONNX_API OfflineTdnnModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineSenseVoiceModelConfig {
  std::string model;
  std::string language;
  bool use_itn = false;
};

struct SHERPA_ONNX_API OfflineDolphinModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineZipformerCtcModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineWenetCtcModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineOmnilingualAsrCtcModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineMedAsrCtcModelConfig {
  std::string model;
};

struct SHERPA_ONNX_API OfflineMoonshineModelConfig {
  std::string preprocessor;
  std::string encoder;
  std::string uncached_decoder;
  std::string cached_decoder;
};

struct SHERPA_ONNX_API OfflineFunASRNanoModelConfig {
  std::string encoder_adaptor;
  std::string llm;
  std::string embedding;
  std::string tokenizer;
  std::string system_prompt = "You are a helpful assistant.";
  std::string user_prompt = "语音转写：";
  int32_t max_new_tokens = 512;
  float temperature = 1e-6f;
  float top_p = 0.8f;
  int32_t seed = 42;
};

struct SHERPA_ONNX_API OfflineModelConfig {
  OfflineTransducerModelConfig transducer;
  OfflineParaformerModelConfig paraformer;
  OfflineNemoEncDecCtcModelConfig nemo_ctc;
  OfflineWhisperModelConfig whisper;
  OfflineTdnnModelConfig tdnn;

  std::string tokens;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";
  std::string model_type;
  std::string modeling_unit = "cjkchar";
  std::string bpe_vocab;
  std::string telespeech_ctc;
  OfflineSenseVoiceModelConfig sense_voice;
  OfflineMoonshineModelConfig moonshine;
  OfflineFireRedAsrModelConfig fire_red_asr;
  OfflineDolphinModelConfig dolphin;
  OfflineZipformerCtcModelConfig zipformer_ctc;
  OfflineCanaryModelConfig canary;
  OfflineWenetCtcModelConfig wenet_ctc;
  OfflineOmnilingualAsrCtcModelConfig omnilingual;
  OfflineMedAsrCtcModelConfig medasr;
  OfflineFunASRNanoModelConfig funasr_nano;
};

struct SHERPA_ONNX_API OfflineLMConfig {
  std::string model;
  float scale = 1.0;
};

struct SHERPA_ONNX_API OfflineRecognizerConfig {
  FeatureConfig feat_config;
  OfflineModelConfig model_config;
  OfflineLMConfig lm_config;

  std::string decoding_method = "greedy_search";
  int32_t max_active_paths = 4;

  std::string hotwords_file;

  float hotwords_score = 1.5;
  std::string rule_fsts;
  std::string rule_fars;
  float blank_penalty = 0;
  HomophoneReplacerConfig hr;
};

struct SHERPA_ONNX_API OfflineRecognizerResult {
  std::string text;
  std::vector<float> timestamps;
  std::vector<std::string> tokens;
  std::string json;
  std::string lang;
  std::string emotion;
  std::string event;

  // non-empty only for TDT models
  std::vector<float> durations;
};

class SHERPA_ONNX_API OfflineStream
    : public MoveOnly<OfflineStream, SherpaOnnxOfflineStream> {
 public:
  explicit OfflineStream(const SherpaOnnxOfflineStream *p);

  void AcceptWaveform(int32_t sample_rate, const float *samples,
                      int32_t n) const;

  void Destroy(const SherpaOnnxOfflineStream *p) const;
};

class SHERPA_ONNX_API OfflineRecognizer
    : public MoveOnly<OfflineRecognizer, SherpaOnnxOfflineRecognizer> {
 public:
  static OfflineRecognizer Create(const OfflineRecognizerConfig &config);

  void Destroy(const SherpaOnnxOfflineRecognizer *p) const;

  OfflineStream CreateStream() const;

  OfflineStream CreateStream(const std::string &hotwords) const;

  void Decode(const OfflineStream *s) const;

  void Decode(const OfflineStream *ss, int32_t n) const;

  OfflineRecognizerResult GetResult(const OfflineStream *s) const;

  // For unreal engine, please use this function
  // See also https://github.com/k2-fsa/sherpa-onnx/discussions/1960
  std::shared_ptr<OfflineRecognizerResult> GetResultPtr(
      const OfflineStream *s) const;

  void SetConfig(const OfflineRecognizerConfig &config) const;

 private:
  explicit OfflineRecognizer(const SherpaOnnxOfflineRecognizer *p);
};

// ============================================================================
// Non-streaming TTS
// ============================================================================
struct OfflineTtsVitsModelConfig {
  std::string model;
  std::string lexicon;
  std::string tokens;
  std::string data_dir;
  std::string dict_dir;  // unused

  float noise_scale = 0.667;
  float noise_scale_w = 0.8;
  float length_scale = 1.0;  // < 1, faster in speed; > 1, slower in speed
};

struct OfflineTtsMatchaModelConfig {
  std::string acoustic_model;
  std::string vocoder;
  std::string lexicon;
  std::string tokens;
  std::string data_dir;
  std::string dict_dir;  // unused

  float noise_scale = 0.667;
  float length_scale = 1.0;  // < 1, faster in speed; > 1, slower in speed
};

struct OfflineTtsKokoroModelConfig {
  std::string model;
  std::string voices;
  std::string tokens;
  std::string data_dir;
  std::string dict_dir;  // unused
  std::string lexicon;
  std::string lang;

  float length_scale = 1.0;  // < 1, faster in speed; > 1, slower in speed
};

struct OfflineTtsKittenModelConfig {
  std::string model;
  std::string voices;
  std::string tokens;
  std::string data_dir;

  float length_scale = 1.0;  // < 1, faster in speed; > 1, slower in speed
};

struct OfflineTtsZipvoiceModelConfig {
  std::string tokens;
  std::string encoder;
  std::string decoder;
  std::string vocoder;
  std::string data_dir;
  std::string lexicon;

  float feat_scale = 0.1;
  float t_shift = 0.5;
  float target_rms = 0.1;
  float guidance_scale = 1.0;
};

struct OfflineTtsModelConfig {
  OfflineTtsVitsModelConfig vits;
  OfflineTtsMatchaModelConfig matcha;
  OfflineTtsKokoroModelConfig kokoro;
  OfflineTtsKittenModelConfig kitten;
  OfflineTtsZipvoiceModelConfig zipvoice;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";
};

struct OfflineTtsConfig {
  OfflineTtsModelConfig model;
  std::string rule_fsts;
  std::string rule_fars;
  int32_t max_num_sentences = 1;
  float silence_scale = 0.2;
};

struct GeneratedAudio {
  std::vector<float> samples;  // in the range [-1, 1]
  int32_t sample_rate;
};

// Return 1 to continue generating
// Return 0 to stop generating
using OfflineTtsCallback = int32_t (*)(const float *samples,
                                       int32_t num_samples, float progress,
                                       void *arg);

class SHERPA_ONNX_API OfflineTts
    : public MoveOnly<OfflineTts, SherpaOnnxOfflineTts> {
 public:
  static OfflineTts Create(const OfflineTtsConfig &config);

  void Destroy(const SherpaOnnxOfflineTts *p) const;

  // Return the sample rate of the generated audio
  int32_t SampleRate() const;

  // Number of supported speakers.
  // If it supports only a single speaker, then it return 0 or 1.
  int32_t NumSpeakers() const;

  // @param text A string containing words separated by spaces
  // @param sid Speaker ID. Used only for multi-speaker models, e.g., models
  //            trained using the VCTK dataset. It is not used for
  //            single-speaker models, e.g., models trained using the ljspeech
  //            dataset.
  // @param speed The speed for the generated speech. E.g., 2 means 2x faster.
  // @param callback If not NULL, it is called whenever config.max_num_sentences
  //                 sentences have been processed. The callback is called in
  //                 the current thread.
  GeneratedAudio Generate(const std::string &text, int32_t sid = 0,
                          float speed = 1.0,
                          OfflineTtsCallback callback = nullptr,
                          void *arg = nullptr) const;

  // Like Generate, but return a smart pointer.
  //
  // See also https://github.com/k2-fsa/sherpa-onnx/issues/2347
  std::shared_ptr<GeneratedAudio> Generate2(
      const std::string &text, int32_t sid = 0, float speed = 1.0,
      OfflineTtsCallback callback = nullptr, void *arg = nullptr) const;

 private:
  explicit OfflineTts(const SherpaOnnxOfflineTts *p);
};

// ============================================================
// For Keyword Spotter
// ============================================================

struct KeywordResult {
  std::string keyword;
  std::vector<std::string> tokens;
  std::vector<float> timestamps;
  float start_time;
  std::string json;
};

struct KeywordSpotterConfig {
  FeatureConfig feat_config;
  OnlineModelConfig model_config;
  int32_t max_active_paths = 4;
  int32_t num_trailing_blanks = 1;
  float keywords_score = 1.0f;
  float keywords_threshold = 0.25f;
  std::string keywords_file;
};

class SHERPA_ONNX_API KeywordSpotter
    : public MoveOnly<KeywordSpotter, SherpaOnnxKeywordSpotter> {
 public:
  static KeywordSpotter Create(const KeywordSpotterConfig &config);

  void Destroy(const SherpaOnnxKeywordSpotter *p) const;

  OnlineStream CreateStream() const;

  OnlineStream CreateStream(const std::string &keywords) const;

  bool IsReady(const OnlineStream *s) const;

  void Decode(const OnlineStream *s) const;

  void Decode(const OnlineStream *ss, int32_t n) const;

  void Reset(const OnlineStream *s) const;

  KeywordResult GetResult(const OnlineStream *s) const;

 private:
  explicit KeywordSpotter(const SherpaOnnxKeywordSpotter *p);
};

struct OfflineSpeechDenoiserGtcrnModelConfig {
  std::string model;
};

struct OfflineSpeechDenoiserModelConfig {
  OfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  int32_t num_threads = 1;
  int32_t debug = false;
  std::string provider = "cpu";
};

struct OfflineSpeechDenoiserConfig {
  OfflineSpeechDenoiserModelConfig model;
};

struct DenoisedAudio {
  std::vector<float> samples;  // in the range [-1, 1]
  int32_t sample_rate;
};

class SHERPA_ONNX_API OfflineSpeechDenoiser
    : public MoveOnly<OfflineSpeechDenoiser, SherpaOnnxOfflineSpeechDenoiser> {
 public:
  static OfflineSpeechDenoiser Create(
      const OfflineSpeechDenoiserConfig &config);

  void Destroy(const SherpaOnnxOfflineSpeechDenoiser *p) const;

  DenoisedAudio Run(const float *samples, int32_t n, int32_t sample_rate) const;

  int32_t GetSampleRate() const;

 private:
  explicit OfflineSpeechDenoiser(const SherpaOnnxOfflineSpeechDenoiser *p);
};

// ==============================
// VAD
// ==============================

struct SileroVadModelConfig {
  std::string model;
  float threshold = 0.5;
  float min_silence_duration = 0.5;
  float min_speech_duration = 0.25;
  int32_t window_size = 512;
  float max_speech_duration = 20;
};

struct TenVadModelConfig {
  std::string model;
  float threshold = 0.5;
  float min_silence_duration = 0.5;
  float min_speech_duration = 0.25;
  int32_t window_size = 256;
  float max_speech_duration = 20;
};

struct VadModelConfig {
  SileroVadModelConfig silero_vad;
  TenVadModelConfig ten_vad;

  int32_t sample_rate = 16000;
  int32_t num_threads = 1;
  std::string provider = "cpu";
  bool debug = false;
};

struct SpeechSegment {
  int32_t start;
  std::vector<float> samples;
};

class SHERPA_ONNX_API CircularBuffer
    : public MoveOnly<CircularBuffer, SherpaOnnxCircularBuffer> {
 public:
  static CircularBuffer Create(int32_t capacity);

  void Destroy(const SherpaOnnxCircularBuffer *p) const;

  void Push(const float *p, int32_t n) const;

  std::vector<float> Get(int32_t start_index, int32_t n) const;

  void Pop(int32_t n) const;

  int32_t Size() const;

  int32_t Head() const;

  void Reset() const;

 private:
  explicit CircularBuffer(const SherpaOnnxCircularBuffer *p);
};

class SHERPA_ONNX_API VoiceActivityDetector
    : public MoveOnly<VoiceActivityDetector, SherpaOnnxVoiceActivityDetector> {
 public:
  static VoiceActivityDetector Create(const VadModelConfig &config,
                                      float buffer_size_in_seconds);

  void Destroy(const SherpaOnnxVoiceActivityDetector *p) const;

  void AcceptWaveform(const float *samples, int32_t n) const;

  bool IsEmpty() const;

  bool IsDetected() const;

  void Pop() const;

  void Clear() const;

  SpeechSegment Front() const;

  // For unreal engine, please use this function
  // See also https://github.com/k2-fsa/sherpa-onnx/discussions/1960
  std::shared_ptr<SpeechSegment> FrontPtr() const;

  void Reset() const;

  void Flush() const;

 private:
  explicit VoiceActivityDetector(const SherpaOnnxVoiceActivityDetector *p);
};

class SHERPA_ONNX_API LinearResampler
    : public MoveOnly<LinearResampler, SherpaOnnxLinearResampler> {
 public:
  LinearResampler() = default;
  static LinearResampler Create(int32_t samp_rate_in_hz,
                                int32_t samp_rate_out_hz,
                                float filter_cutoff_hz, int32_t num_zeros);

  void Destroy(const SherpaOnnxLinearResampler *p) const;

  void Reset() const;

  std::vector<float> Resample(const float *input, int32_t input_dim,
                              bool flush) const;

  int32_t GetInputSamplingRate() const;
  int32_t GetOutputSamplingRate() const;

 private:
  explicit LinearResampler(const SherpaOnnxLinearResampler *p);
};

SHERPA_ONNX_API std::string GetVersionStr();
SHERPA_ONNX_API std::string GetGitSha1();
SHERPA_ONNX_API std::string GetGitDate();
SHERPA_ONNX_API bool FileExists(const std::string &filename);

// ============================================================================
// Offline Punctuation
// ============================================================================
struct OfflinePunctuationModelConfig {
  std::string ct_transformer;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";
};

struct OfflinePunctuationConfig {
  OfflinePunctuationModelConfig model;
};

class SHERPA_ONNX_API OfflinePunctuation
    : public MoveOnly<OfflinePunctuation, SherpaOnnxOfflinePunctuation> {
 public:
  static OfflinePunctuation Create(const OfflinePunctuationConfig &config);

  void Destroy(const SherpaOnnxOfflinePunctuation *p) const;

  // Add punctuations to the input text and return it.
  std::string AddPunctuation(const std::string &text) const;

 private:
  explicit OfflinePunctuation(const SherpaOnnxOfflinePunctuation *p);
};

// ============================================================================
// Online Punctuation
// ============================================================================
struct OnlinePunctuationModelConfig {
  std::string cnn_bilstm;
  std::string bpe_vocab;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";
};

struct OnlinePunctuationConfig {
  OnlinePunctuationModelConfig model;
};

class SHERPA_ONNX_API OnlinePunctuation
    : public MoveOnly<OnlinePunctuation, SherpaOnnxOnlinePunctuation> {
 public:
  static OnlinePunctuation Create(const OnlinePunctuationConfig &config);

  void Destroy(const SherpaOnnxOnlinePunctuation *p) const;

  // Add punctuations to the input text and return it.
  std::string AddPunctuation(const std::string &text) const;

 private:
  explicit OnlinePunctuation(const SherpaOnnxOnlinePunctuation *p);
};

// ============================================================================
// Audio tagging
// ============================================================================
struct OfflineZipformerAudioTaggingModelConfig {
  std::string model;
};

struct AudioTaggingModelConfig {
  OfflineZipformerAudioTaggingModelConfig zipformer;
  std::string ced;
  int32_t num_threads = 1;
  bool debug = false;  // true to print debug information of the model
  std::string provider = "cpu";
};

struct AudioTaggingConfig {
  AudioTaggingModelConfig model;
  std::string labels;
  int32_t top_k = 5;
};

struct AudioEvent {
  std::string name;
  int32_t index;
  float prob;
};

class SHERPA_ONNX_API AudioTagging
    : public MoveOnly<AudioTagging, SherpaOnnxAudioTagging> {
 public:
  static AudioTagging Create(const AudioTaggingConfig &config);

  void Destroy(const SherpaOnnxAudioTagging *p) const;

  OfflineStream CreateStream() const;
  // when top_k is -1, it uses the top_k from config.top_k
  // when top_k is > 0, config.top_k is ignored
  std::vector<AudioEvent> Compute(const OfflineStream *s, int32_t top_k = -1);

  // For unreal engine, please use this function
  // See also https://github.com/k2-fsa/sherpa-onnx/discussions/1960
  std::shared_ptr<std::vector<AudioEvent>> ComputePtr(const OfflineStream *s,
                                                      int32_t top_k = -1);

 private:
  explicit AudioTagging(const SherpaOnnxAudioTagging *p);
};

}  // namespace sherpa_onnx::cxx

#endif  // SHERPA_ONNX_C_API_CXX_API_H_
