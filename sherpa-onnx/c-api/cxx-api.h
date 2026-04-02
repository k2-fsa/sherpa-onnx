// sherpa-onnx/c-api/cxx-api.h
//
// Copyright (c)  2024  Xiaomi Corporation
/**
 * @file cxx-api.h
 * @brief Public C++ wrapper for the sherpa-onnx C API.
 *
 * This header provides a lightweight C++ interface on top of `c-api.h`. The
 * wrapper follows a few simple design rules:
 *
 * - Configuration objects are plain structs with `std::string`,
 *   `std::vector`, and default values
 * - Runtime handles are move-only RAII classes that automatically release the
 *   underlying C handle
 * - Result objects are copied into standard C++ containers so callers do not
 *   need to manage C-allocated memory manually
 * - The API mirrors the C API closely, while offering a more idiomatic C++
 *   surface
 *
 * Major feature families available in this file:
 *
 * - Streaming ASR
 * - Non-streaming ASR
 * - Non-streaming TTS
 * - Keyword spotting
 * - Offline and online speech enhancement
 * - VAD and circular buffering
 * - Linear resampling
 * - Version/file/WAVE helpers
 * - Offline and online punctuation
 * - Audio tagging
 *
 * Typical usage pattern:
 *
 * 1. Fill a config struct
 * 2. Create the corresponding RAII wrapper with `Class::Create(...)`
 * 3. Check `wrapper.Get()` for success
 * 4. Feed audio or text, run inference, and retrieve results as C++ objects
 * 5. Let destructors clean up automatically
 *
 * Example programs are available in `cxx-api-examples/` and show concrete model
 * packages and end-to-end usage.
 */
#ifndef SHERPA_ONNX_C_API_CXX_API_H_
#define SHERPA_ONNX_C_API_CXX_API_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"

namespace sherpa_onnx::cxx {

// ============================================================================
// Streaming ASR
// ============================================================================
/** @brief Streaming transducer model files. */
struct OnlineTransducerModelConfig {
  /** Encoder ONNX model. */
  std::string encoder;
  /** Decoder ONNX model. */
  std::string decoder;
  /** Joiner ONNX model. */
  std::string joiner;
};

/** @brief Streaming Paraformer model files. */
struct OnlineParaformerModelConfig {
  /** Encoder ONNX model. */
  std::string encoder;
  /** Decoder ONNX model. */
  std::string decoder;
};

/** @brief Streaming Zipformer2 CTC model file. */
struct OnlineZipformer2CtcModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Streaming NeMo CTC model file. */
struct OnlineNemoCtcModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Streaming T-One CTC model file. */
struct OnlineToneCtcModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/**
 * @brief Acoustic model configuration for streaming ASR.
 *
 * Configure exactly one model family. If multiple model families are set, one
 * of them will be chosen and the choice is implementation-defined.
 *
 * Example using
 * `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`:
 *
 * @code
 * OnlineModelConfig model;
 * model.transducer.encoder =
 *     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
 *     "encoder-epoch-99-avg-1.int8.onnx";
 * model.transducer.decoder =
 *     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
 *     "decoder-epoch-99-avg-1.onnx";
 * model.transducer.joiner =
 *     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
 *     "joiner-epoch-99-avg-1.int8.onnx";
 * model.tokens =
 *     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt";
 * model.num_threads = 1;
 * @endcode
 */
struct OnlineModelConfig {
  /** Streaming transducer configuration. */
  OnlineTransducerModelConfig transducer;
  /** Streaming Paraformer configuration. */
  OnlineParaformerModelConfig paraformer;
  /** Streaming Zipformer2 CTC configuration. */
  OnlineZipformer2CtcModelConfig zipformer2_ctc;
  /** Streaming NeMo CTC configuration. */
  OnlineNemoCtcModelConfig nemo_ctc;
  /** Streaming T-One CTC configuration. */
  OnlineToneCtcModelConfig t_one_ctc;
  /** Token file path. */
  std::string tokens;
  /** Number of inference threads. */
  int32_t num_threads = 1;
  /** Execution provider such as `"cpu"`. */
  std::string provider = "cpu";
  /** Enable verbose debug logging. */
  bool debug = false;
  /** Optional explicit model type hint. */
  std::string model_type;
  /** Modeling unit such as `"cjkchar"` or `"bpe"`. */
  std::string modeling_unit = "cjkchar";
  /** Optional BPE vocabulary. */
  std::string bpe_vocab;
  /** Optional in-memory token content. If non-empty, it is used instead of a
   * file. */
  std::string tokens_buf;
};

/** @brief Feature extraction settings shared by ASR and KWS wrappers. */
struct FeatureConfig {
  /** Input sample rate in Hz. */
  int32_t sample_rate = 16000;
  /** Number of features per frame. */
  int32_t feature_dim = 80;
};

/** @brief Decoder graph configuration for online CTC + FST decoding. */
struct OnlineCtcFstDecoderConfig {
  /** FST graph file. */
  std::string graph;
  /** Maximum number of active states during search. */
  int32_t max_active = 3000;
};

/** @brief Homophone replacement resources used by some Chinese ASR setups. */
struct HomophoneReplacerConfig {
  /** Reserved field. Currently unused by the wrapper. */
  std::string dict_dir;
  /** Lexicon file used by the replacer. */
  std::string lexicon;
  /** Rule FST file used for replacement. */
  std::string rule_fsts;
};

/**
 * @brief Configuration for streaming ASR.
 *
 * Example:
 *
 * @code
 * OnlineRecognizerConfig config;
 * config.model_config.transducer.encoder =
 *     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
 *     "encoder-epoch-99-avg-1.int8.onnx";
 * config.model_config.transducer.decoder =
 *     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
 *     "decoder-epoch-99-avg-1.onnx";
 * config.model_config.transducer.joiner =
 *     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/"
 *     "joiner-epoch-99-avg-1.int8.onnx";
 * config.model_config.tokens =
 *     "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt";
 * config.model_config.num_threads = 1;
 * config.hr.lexicon = "./lexicon.txt";
 * config.hr.rule_fsts = "./replace.fst";
 * @endcode
 */
struct OnlineRecognizerConfig {
  /** Feature extraction configuration. */
  FeatureConfig feat_config;
  /** Acoustic model configuration. */
  OnlineModelConfig model_config;

  /** Decoding method such as `"greedy_search"` or `"modified_beam_search"`. */
  std::string decoding_method = "greedy_search";

  /** Maximum number of active paths for beam-search-style decoding. */
  int32_t max_active_paths = 4;

  /** Enable endpoint detection. */
  bool enable_endpoint = false;

  /** Endpointing rule 1 trailing silence threshold in seconds. */
  float rule1_min_trailing_silence = 2.4;

  /** Endpointing rule 2 trailing silence threshold in seconds. */
  float rule2_min_trailing_silence = 1.2;

  /** Endpointing rule 3 minimum utterance length in seconds. */
  float rule3_min_utterance_length = 20;

  /** Optional hotword file. */
  std::string hotwords_file;

  /** Hotword boost score. */
  float hotwords_score = 1.5;

  /** Optional CTC+FST decoder configuration. */
  OnlineCtcFstDecoderConfig ctc_fst_decoder_config;
  /** Optional ITN rule FST archive. */
  std::string rule_fsts;
  /** Optional ITN rule FAR archive. */
  std::string rule_fars;
  /** Optional blank penalty applied during decoding. */
  float blank_penalty = 0;

  /** Optional in-memory hotword definitions. */
  std::string hotwords_buf;
  /** Optional homophone replacement configuration. */
  HomophoneReplacerConfig hr;
};

/** @brief Current streaming ASR result copied into C++ containers. */
struct OnlineRecognizerResult {
  /** Decoded text. */
  std::string text;
  /** Token sequence. */
  std::vector<std::string> tokens;
  /** Per-token timestamps in seconds. */
  std::vector<float> timestamps;
  /** JSON representation of the result. */
  std::string json;
};

/** @brief Mono PCM waveform used by the helper I/O functions. */
struct Wave {
  /** Samples normalized to `[-1, 1]`. */
  std::vector<float> samples;
  /** Sample rate in Hz. */
  int32_t sample_rate = 0;
};

/**
 * @brief Read a mono WAVE file into a C++ value object.
 *
 * On failure, the returned wave has `samples.empty() == true`.
 *
 * @param filename Input WAVE filename.
 * @return Decoded wave data.
 */
SHERPA_ONNX_API Wave ReadWave(const std::string &filename);

/**
 * @brief Write a mono WAVE file from a C++ value object.
 *
 * @param filename Output filename.
 * @param wave PCM samples and sample rate to write.
 * @return `true` on success; `false` on failure.
 */
SHERPA_ONNX_API bool WriteWave(const std::string &filename, const Wave &wave);

/**
 * @brief Base class for move-only RAII wrappers around C handles.
 *
 * Derived classes implement `Destroy(const T *) const` and inherit automatic
 * destruction, `Get()`, and `Release()`.
 */
template <typename Derived, typename T>
class SHERPA_ONNX_API MoveOnly {
 public:
  /** @brief Construct an empty wrapper. */
  MoveOnly() = default;
  /** @brief Construct a wrapper from a raw C handle. */
  explicit MoveOnly(const T *p) : p_(p) {}

  /** @brief Destroy the wrapped handle if present. */
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

  /** @brief Return the wrapped raw pointer without transferring ownership. */
  const T *Get() const { return p_; }

  /** @brief Release ownership of the wrapped raw pointer. */
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
  /** @brief Wrap an existing C online stream handle. */
  explicit OnlineStream(const SherpaOnnxOnlineStream *p);

  /** @brief Append audio samples to the stream. */
  void AcceptWaveform(int32_t sample_rate, const float *samples,
                      int32_t n) const;

  /** @brief Indicate that no more input audio will be provided. */
  void InputFinished() const;

  /** @brief Set a per-stream string option. */
  void SetOption(const char *key, const char *value) const;
  /** @brief Get a per-stream string option. */
  const char *GetOption(const char *key) const;
  /** @brief Check whether a per-stream option exists. */
  int32_t HasOption(const char *key) const;

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOnlineStream *p) const;
};

/**
 * @brief RAII wrapper for a streaming recognizer.
 *
 * Example:
 *
 * @code
 * OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
 * OnlineStream stream = recognizer.CreateStream();
 * stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
 *                       wave.samples.size());
 * stream.InputFinished();
 * while (recognizer.IsReady(&stream)) {
 *   recognizer.Decode(&stream);
 * }
 * auto result = recognizer.GetResult(&stream);
 * @endcode
 */
class SHERPA_ONNX_API OnlineRecognizer
    : public MoveOnly<OnlineRecognizer, SherpaOnnxOnlineRecognizer> {
 public:
  /** @brief Create a streaming recognizer from a config struct. */
  static OnlineRecognizer Create(const OnlineRecognizerConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOnlineRecognizer *p) const;

  /** @brief Create a stream that uses the recognizer's configured hotwords. */
  OnlineStream CreateStream() const;

  /** @brief Create a stream with inline hotwords. */
  OnlineStream CreateStream(const std::string &hotwords) const;

  /** @brief Check whether the given stream has enough data to decode. */
  bool IsReady(const OnlineStream *s) const;

  /** @brief Decode one ready stream. */
  void Decode(const OnlineStream *s) const;

  /** @brief Decode multiple ready streams in parallel. */
  void Decode(const OnlineStream *ss, int32_t n) const;

  /** @brief Return the current recognition result for a stream. */
  OnlineRecognizerResult GetResult(const OnlineStream *s) const;

  /** @brief Reset a stream after endpointing or utterance completion. */
  void Reset(const OnlineStream *s) const;

  /** @brief Check whether endpointing has triggered for a stream. */
  bool IsEndpoint(const OnlineStream *s) const;

 private:
  explicit OnlineRecognizer(const SherpaOnnxOnlineRecognizer *p);
};

// ============================================================================
// Non-streaming ASR
// ============================================================================
/** @brief Offline transducer model files. */
struct OfflineTransducerModelConfig {
  /** Encoder ONNX model. */
  std::string encoder;
  /** Decoder ONNX model. */
  std::string decoder;
  /** Joiner ONNX model. */
  std::string joiner;
};

/** @brief Offline Paraformer model file. */
struct OfflineParaformerModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Offline NeMo EncDec CTC model file. */
struct OfflineNemoEncDecCtcModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Offline Whisper model configuration. */
struct OfflineWhisperModelConfig {
  /** Encoder ONNX model. */
  std::string encoder;
  /** Decoder ONNX model. */
  std::string decoder;
  /** Whisper language string such as `"en"` or `"zh"`. */
  std::string language;
  /** Task such as `"transcribe"` or `"translate"`. */
  std::string task = "transcribe";
  /** Optional tail paddings in samples. */
  int32_t tail_paddings = -1;
  /** Enable token timestamps in the result. */
  bool enable_token_timestamps = false;
  /** Enable segment timestamps in the result JSON. */
  bool enable_segment_timestamps = false;
};

/** @brief Offline Canary model configuration. */
struct OfflineCanaryModelConfig {
  /** Encoder ONNX model. */
  std::string encoder;
  /** Decoder ONNX model. */
  std::string decoder;
  /** Source language code. */
  std::string src_lang;
  /** Target language code. */
  std::string tgt_lang;
  /** Whether punctuation/casing is enabled by the model. */
  bool use_pnc = true;
};

/** @brief Offline Cohere Transcribe model configuration. */
struct OfflineCohereTranscribeModelConfig {
  /** Encoder ONNX model. */
  std::string encoder;
  /** Decoder ONNX model. */
  std::string decoder;
  /** Cohere language string such as `"en"` or `"zh"`. */
  std::string language;
  /** Whether punctuation is enabled by the model. */
  bool use_punct = true;
  /** Whether inverse text normalization is enabled. */
  bool use_itn = true;
};

/** @brief Offline FireRed ASR model files. */
struct OfflineFireRedAsrModelConfig {
  /** Encoder ONNX model. */
  std::string encoder;
  /** Decoder ONNX model. */
  std::string decoder;
};

/** @brief Offline FireRed ASR CTC model file. */
struct OfflineFireRedAsrCtcModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Offline TDNN model file. */
struct OfflineTdnnModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Offline SenseVoice model configuration. */
struct OfflineSenseVoiceModelConfig {
  /** Model ONNX file. */
  std::string model;
  /** Language hint. */
  std::string language;
  /** Enable inverse text normalization. */
  bool use_itn = false;
};

/** @brief Offline Dolphin model file. */
struct OfflineDolphinModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Offline Zipformer CTC model file. */
struct OfflineZipformerCtcModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Offline WeNet CTC model file. */
struct OfflineWenetCtcModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Offline omnilingual ASR CTC model file. */
struct OfflineOmnilingualAsrCtcModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Offline MedASR CTC model file. */
struct OfflineMedAsrCtcModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief Offline Moonshine model configuration. */
struct OfflineMoonshineModelConfig {
  /** Preprocessor model file. */
  std::string preprocessor;
  /** Encoder model file. */
  std::string encoder;
  /** Uncached decoder model file. */
  std::string uncached_decoder;
  /** Cached decoder model file. */
  std::string cached_decoder;
  /** Merged decoder model file. */
  std::string merged_decoder;
};

/** @brief Offline FunASR Nano model configuration. */
struct OfflineFunASRNanoModelConfig {
  /** Encoder adaptor model file. */
  std::string encoder_adaptor;
  /** LLM model file. */
  std::string llm;
  /** Embedding model file. */
  std::string embedding;
  /** Tokenizer file. */
  std::string tokenizer;
  /** System prompt passed to the model. */
  std::string system_prompt = "You are a helpful assistant.";
  /** User prompt prefix passed to the model. */
  std::string user_prompt = "语音转写：";
  /** Maximum number of generated tokens. */
  int32_t max_new_tokens = 512;
  /** Sampling temperature. */
  float temperature = 1e-6f;
  /** Top-p sampling parameter. */
  float top_p = 0.8f;
  /** Random seed. */
  int32_t seed = 42;
  /** Language hint. */
  std::string language;
  /** Enable inverse text normalization. */
  bool itn = true;
  /** Optional hotwords string. */
  std::string hotwords;
};

/** @brief Offline Qwen3-ASR model configuration. */
struct OfflineQwen3ASRModelConfig {
  /** Conv-frontend ONNX model file. */
  std::string conv_frontend;
  /** Encoder ONNX model file. */
  std::string encoder;
  /** Decoder ONNX model file (KV cache). */
  std::string decoder;
  /** Tokenizer directory (e.g. containing `vocab.json`). */
  std::string tokenizer;
  /** Optional comma-separated hotwords (UTF-8, ASCII ','), e.g. @c
   * "foo,bar,baz". */
  std::string hotwords;
  /** Maximum total sequence length supported by the model. */
  int32_t max_total_len = 512;
  /** Maximum number of new tokens to generate. */
  int32_t max_new_tokens = 128;
  /** Sampling temperature. */
  float temperature = 1e-6f;
  /** Top-p (nucleus) sampling parameter. */
  float top_p = 0.8f;
  /** Random seed for reproducible sampling. */
  int32_t seed = 42;
};

/**
 * @brief Acoustic model configuration for offline ASR.
 *
 * Configure exactly one model family. If multiple model families are set, one
 * is chosen and the choice is implementation-defined.
 */
struct OfflineModelConfig {
  /** Offline transducer configuration. */
  OfflineTransducerModelConfig transducer;
  /** Offline Paraformer configuration. */
  OfflineParaformerModelConfig paraformer;
  /** Offline NeMo CTC configuration. */
  OfflineNemoEncDecCtcModelConfig nemo_ctc;
  /** Offline Whisper configuration. */
  OfflineWhisperModelConfig whisper;
  /** Offline TDNN configuration. */
  OfflineTdnnModelConfig tdnn;

  /** Token file. */
  std::string tokens;
  /** Number of inference threads. */
  int32_t num_threads = 1;
  /** Enable verbose debug logging. */
  bool debug = false;
  /** Execution provider such as `"cpu"`. */
  std::string provider = "cpu";
  /** Optional explicit model type hint. */
  std::string model_type;
  /** Modeling unit such as `"cjkchar"` or `"bpe"`. */
  std::string modeling_unit = "cjkchar";
  /** Optional BPE vocabulary. */
  std::string bpe_vocab;
  /** Telespeech CTC model file. */
  std::string telespeech_ctc;
  /** SenseVoice configuration. */
  OfflineSenseVoiceModelConfig sense_voice;
  /** Moonshine configuration. */
  OfflineMoonshineModelConfig moonshine;
  /** FireRed transducer configuration. */
  OfflineFireRedAsrModelConfig fire_red_asr;
  /** Dolphin configuration. */
  OfflineDolphinModelConfig dolphin;
  /** Zipformer CTC configuration. */
  OfflineZipformerCtcModelConfig zipformer_ctc;
  /** Canary configuration. */
  OfflineCanaryModelConfig canary;
  /** WeNet CTC configuration. */
  OfflineWenetCtcModelConfig wenet_ctc;
  /** Omnilingual ASR configuration. */
  OfflineOmnilingualAsrCtcModelConfig omnilingual;
  /** MedASR configuration. */
  OfflineMedAsrCtcModelConfig medasr;
  /** FunASR Nano configuration. */
  OfflineFunASRNanoModelConfig funasr_nano;
  /** FireRed CTC configuration. */
  OfflineFireRedAsrCtcModelConfig fire_red_asr_ctc;
  /** Qwen3-ASR configuration. */
  OfflineQwen3ASRModelConfig qwen3_asr;
  /** Cohere Transcribe configuration. */
  OfflineCohereTranscribeModelConfig cohere_transcribe;
};

/** @brief Optional language-model rescoring configuration for offline ASR. */
struct OfflineLMConfig {
  /** LM model file. */
  std::string model;
  /** LM scale. */
  float scale = 1.0;
};

/**
 * @brief Configuration for offline ASR.
 *
 * Example using SenseVoice:
 *
 * @code
 * OfflineRecognizerConfig config;
 * config.model_config.sense_voice.model =
 *     "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/model.int8.onnx";
 * config.model_config.sense_voice.language = "auto";
 * config.model_config.sense_voice.use_itn = true;
 * config.model_config.tokens =
 *     "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/tokens.txt";
 * config.model_config.num_threads = 1;
 * @endcode
 *
 * Example using Parakeet TDT v2:
 *
 * @code
 * OfflineRecognizerConfig config;
 * config.model_config.transducer.encoder =
 *     "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx";
 * config.model_config.transducer.decoder =
 *     "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx";
 * config.model_config.transducer.joiner =
 *     "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx";
 * config.model_config.tokens =
 *     "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt";
 * config.model_config.model_type = "nemo_transducer";
 * config.model_config.num_threads = 1;
 * @endcode
 */
struct OfflineRecognizerConfig {
  /** Feature extraction configuration. */
  FeatureConfig feat_config;
  /** Acoustic model configuration. */
  OfflineModelConfig model_config;
  /** Optional LM configuration. */
  OfflineLMConfig lm_config;

  /** Decoding method such as `"greedy_search"` or `"modified_beam_search"`. */
  std::string decoding_method = "greedy_search";
  /** Maximum number of active paths for beam-search-style decoding. */
  int32_t max_active_paths = 4;

  /** Optional hotword file. */
  std::string hotwords_file;

  /** Hotword boost score. */
  float hotwords_score = 1.5;
  /** Optional ITN rule FST archive. */
  std::string rule_fsts;
  /** Optional ITN rule FAR archive. */
  std::string rule_fars;
  /** Optional blank penalty applied during decoding. */
  float blank_penalty = 0;
  /** Optional homophone replacement configuration. */
  HomophoneReplacerConfig hr;
};

/** @brief Offline ASR result copied into C++ containers. */
struct OfflineRecognizerResult {
  /** Decoded text. */
  std::string text;
  /** Per-token timestamps in seconds when available. */
  std::vector<float> timestamps;
  /** Token sequence. */
  std::vector<std::string> tokens;
  /** JSON representation of the result. */
  std::string json;
  /** Detected language when provided by the model. */
  std::string lang;
  /** Detected emotion when provided by the model. */
  std::string emotion;
  /** Detected event when provided by the model. */
  std::string event;

  /** Non-empty only for TDT-style models. */
  std::vector<float> durations;
};

/** @brief RAII wrapper for an offline decoding stream. */
class SHERPA_ONNX_API OfflineStream
    : public MoveOnly<OfflineStream, SherpaOnnxOfflineStream> {
 public:
  /** @brief Wrap an existing C offline stream handle. */
  explicit OfflineStream(const SherpaOnnxOfflineStream *p);

  /** @brief Provide the complete waveform for offline decoding. */
  void AcceptWaveform(int32_t sample_rate, const float *samples,
                      int32_t n) const;

  /** @brief Set a per-stream string option. */
  void SetOption(const char *key, const char *value) const;
  /** @brief Get a per-stream string option. */
  const char *GetOption(const char *key) const;
  /** @brief Check whether a per-stream option exists. */
  int32_t HasOption(const char *key) const;

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOfflineStream *p) const;
};

/**
 * @brief RAII wrapper for an offline recognizer.
 *
 * For most offline models, call `AcceptWaveform()` once per stream, then call
 * `Decode()` and `GetResult()`.
 */
class SHERPA_ONNX_API OfflineRecognizer
    : public MoveOnly<OfflineRecognizer, SherpaOnnxOfflineRecognizer> {
 public:
  /** @brief Create an offline recognizer from a config struct. */
  static OfflineRecognizer Create(const OfflineRecognizerConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOfflineRecognizer *p) const;

  /** @brief Create a stream using the recognizer's configured hotwords. */
  OfflineStream CreateStream() const;

  /** @brief Create a stream with inline hotwords. */
  OfflineStream CreateStream(const std::string &hotwords) const;

  /** @brief Decode one offline stream. */
  void Decode(const OfflineStream *s) const;

  /** @brief Decode multiple offline streams in parallel. */
  void Decode(const OfflineStream *ss, int32_t n) const;

  /** @brief Return the copied recognition result for one stream. */
  OfflineRecognizerResult GetResult(const OfflineStream *s) const;

  /**
   * @brief Convenience wrapper that returns the result inside a shared pointer.
   *
   * This helper exists mainly for integration environments that prefer owning
   * pointers, such as Unreal Engine.
   */
  std::shared_ptr<OfflineRecognizerResult> GetResultPtr(
      const OfflineStream *s) const;

  /** @brief Update recognizer runtime configuration after creation. */
  void SetConfig(const OfflineRecognizerConfig &config) const;

 private:
  explicit OfflineRecognizer(const SherpaOnnxOfflineRecognizer *p);
};

// ============================================================================
// Non-streaming TTS
// ============================================================================
/** @brief VITS model configuration. */
struct OfflineTtsVitsModelConfig {
  /** Acoustic model file. */
  std::string model;
  /** Lexicon file. */
  std::string lexicon;
  /** Token file. */
  std::string tokens;
  /** Data directory such as `espeak-ng-data`. */
  std::string data_dir;
  /** Reserved field. Currently unused by the wrapper. */
  std::string dict_dir;

  /** VITS noise scale. */
  float noise_scale = 0.667;
  /** VITS noise scale for duration prediction. */
  float noise_scale_w = 0.8;
  /** Length scale. Values < 1 are faster; values > 1 are slower. */
  float length_scale = 1.0;
};

/** @brief Matcha model configuration. */
struct OfflineTtsMatchaModelConfig {
  /** Acoustic model file. */
  std::string acoustic_model;
  /** Vocoder model file. */
  std::string vocoder;
  /** Lexicon file. */
  std::string lexicon;
  /** Token file. */
  std::string tokens;
  /** Data directory such as `espeak-ng-data`. */
  std::string data_dir;
  /** Reserved field. Currently unused by the wrapper. */
  std::string dict_dir;

  /** Matcha noise scale. */
  float noise_scale = 0.667;
  /** Length scale. Values < 1 are faster; values > 1 are slower. */
  float length_scale = 1.0;
};

/** @brief Kokoro model configuration. */
struct OfflineTtsKokoroModelConfig {
  /** Acoustic model file. */
  std::string model;
  /** Voices file. */
  std::string voices;
  /** Token file. */
  std::string tokens;
  /** Data directory such as `espeak-ng-data`. */
  std::string data_dir;
  /** Reserved field. Currently unused by the wrapper. */
  std::string dict_dir;
  /** Optional lexicon file. */
  std::string lexicon;
  /** Language/voice family hint. */
  std::string lang;

  /** Length scale. Values < 1 are faster; values > 1 are slower. */
  float length_scale = 1.0;
};

/** @brief Kitten model configuration. */
struct OfflineTtsKittenModelConfig {
  /** Acoustic model file. */
  std::string model;
  /** Voices file. */
  std::string voices;
  /** Token file. */
  std::string tokens;
  /** Data directory. */
  std::string data_dir;

  /** Length scale. Values < 1 are faster; values > 1 are slower. */
  float length_scale = 1.0;
};

/** @brief ZipVoice model configuration. */
struct OfflineTtsZipvoiceModelConfig {
  /** Token file. */
  std::string tokens;
  /** Encoder model file. */
  std::string encoder;
  /** Decoder model file. */
  std::string decoder;
  /** Vocoder model file. */
  std::string vocoder;
  /** Data directory. */
  std::string data_dir;
  /** Lexicon file. */
  std::string lexicon;

  /** Feature scale. */
  float feat_scale = 0.1;
  /** Time shift. */
  float t_shift = 0.5;
  /** Target RMS. */
  float target_rms = 0.1;
  /** Guidance scale. */
  float guidance_scale = 1.0;
};

/** @brief Pocket TTS model configuration. */
struct OfflineTtsPocketModelConfig {
  /** Flow model file. */
  std::string lm_flow;
  /** Main language model file. */
  std::string lm_main;
  /** Encoder model file. */
  std::string encoder;
  /** Decoder model file. */
  std::string decoder;
  /** Text conditioner model file. */
  std::string text_conditioner;

  /** Vocabulary JSON file. */
  std::string vocab_json;
  /** Token scores JSON file. */
  std::string token_scores_json;
  /** Voice embedding cache size. */
  int32_t voice_embedding_cache_capacity = 50;
};

/** @brief Supertonic model configuration. */
struct OfflineTtsSupertonicModelConfig {
  /** Duration predictor model file. */
  std::string duration_predictor;
  /** Text encoder model file. */
  std::string text_encoder;
  /** Vector estimator model file. */
  std::string vector_estimator;
  /** Vocoder model file. */
  std::string vocoder;
  /** Model metadata JSON. */
  std::string tts_json;
  /** Unicode indexer resource. */
  std::string unicode_indexer;
  /** Voice style resource. */
  std::string voice_style;
};

/**
 * @brief Model configuration for offline TTS.
 *
 * Configure exactly one model family. If multiple model families are set, one
 * is chosen and the choice is implementation-defined.
 */
struct OfflineTtsModelConfig {
  /** VITS configuration. */
  OfflineTtsVitsModelConfig vits;
  /** Matcha configuration. */
  OfflineTtsMatchaModelConfig matcha;
  /** Kokoro configuration. */
  OfflineTtsKokoroModelConfig kokoro;
  /** Kitten configuration. */
  OfflineTtsKittenModelConfig kitten;
  /** ZipVoice configuration. */
  OfflineTtsZipvoiceModelConfig zipvoice;
  /** Pocket configuration. */
  OfflineTtsPocketModelConfig pocket;
  /** Supertonic configuration. */
  OfflineTtsSupertonicModelConfig supertonic;

  /** Number of inference threads. */
  int32_t num_threads = 1;
  /** Enable verbose debug logging. */
  bool debug = false;
  /** Execution provider such as `"cpu"`. */
  std::string provider = "cpu";
};

/** @brief Generation-time options for advanced TTS synthesis. */
struct GenerationConfig {
  /** Silence scale between sentences. */
  float silence_scale = 0.2;
  /** Speech speed. Used only by some models. */
  float speed = 1.0;
  /** Speaker ID for multi-speaker models. */
  int32_t sid = 0;
  /** Reference audio samples for zero-shot or voice-cloning models. */
  std::vector<float> reference_audio;
  /** Sample rate of `reference_audio`. */
  int32_t reference_sample_rate = 0;
  /** Optional reference text. Not all models require it. */
  std::string reference_text;
  /** Number of flow-matching steps when supported. */
  int32_t num_steps = 5;

  /** Model-specific extra attributes serialized to JSON internally. */
  std::unordered_map<std::string, std::string> extra;
};

/** @brief Configuration for offline TTS. */
struct OfflineTtsConfig {
  /** Model configuration. */
  OfflineTtsModelConfig model;
  /** Optional ITN rule FST archive. */
  std::string rule_fsts;
  /** Optional ITN rule FAR archive. */
  std::string rule_fars;
  /** Sentence chunking limit for generation. */
  int32_t max_num_sentences = 1;
  /** Silence scale between generated sentences. */
  float silence_scale = 0.2;
};

/** @brief Generated audio returned by the C++ TTS wrapper. */
struct GeneratedAudio {
  /** Output samples normalized to `[-1, 1]`. */
  std::vector<float> samples;
  /** Output sample rate in Hz. */
  int32_t sample_rate = 0;
};

/**
 * @brief TTS progress callback.
 *
 * Return 1 to continue generating and 0 to stop early.
 */
using OfflineTtsCallback = int32_t (*)(const float *samples,
                                       int32_t num_samples, float progress,
                                       void *arg);

/**
 * @brief RAII wrapper for offline TTS.
 *
 * Example using Pocket TTS:
 *
 * @code
 * OfflineTtsConfig config;
 * config.model.pocket.lm_flow =
 *     "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx";
 * config.model.pocket.lm_main =
 *     "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx";
 * config.model.pocket.encoder =
 *     "./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx";
 * config.model.pocket.decoder =
 *     "./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx";
 * config.model.pocket.text_conditioner =
 *     "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx";
 * config.model.pocket.vocab_json =
 *     "./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json";
 * config.model.pocket.token_scores_json =
 *     "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json";
 * @endcode
 */
class SHERPA_ONNX_API OfflineTts
    : public MoveOnly<OfflineTts, SherpaOnnxOfflineTts> {
 public:
  /** @brief Create an offline TTS engine. */
  static OfflineTts Create(const OfflineTtsConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOfflineTts *p) const;

  /** @brief Return the output sample rate of generated audio. */
  int32_t SampleRate() const;

  /** @brief Return the number of supported speakers. */
  int32_t NumSpeakers() const;

  /**
   * @brief Generate speech using the simple speaker-id and speed interface.
   *
   * This overload mirrors the legacy/simple TTS API. Prefer the
   * `GenerationConfig` overload for new code.
   */
  GeneratedAudio Generate(const std::string &text, int32_t sid = 0,
                          float speed = 1.0,
                          OfflineTtsCallback callback = nullptr,
                          void *arg = nullptr) const;

  /** @brief Generate speech using the advanced generation configuration. */
  GeneratedAudio Generate(const std::string &text,
                          const GenerationConfig &config,
                          OfflineTtsCallback callback = nullptr,
                          void *arg = nullptr) const;

  /** @brief Like Generate(), but returns a shared pointer to the result. */
  std::shared_ptr<GeneratedAudio> Generate2(
      const std::string &text, int32_t sid = 0, float speed = 1.0,
      OfflineTtsCallback callback = nullptr, void *arg = nullptr) const;

  /** @brief Like the advanced Generate() overload, but returns a shared
   * pointer. */
  std::shared_ptr<GeneratedAudio> Generate2(
      const std::string &text, const GenerationConfig &config,
      OfflineTtsCallback callback = nullptr, void *arg = nullptr) const;

 private:
  explicit OfflineTts(const SherpaOnnxOfflineTts *p);
};

// ============================================================
// For Keyword Spotter
// ============================================================

/** @brief Current keyword spotting result copied into C++ containers. */
struct KeywordResult {
  /** Triggered keyword text. */
  std::string keyword;
  /** Decoded token sequence. */
  std::vector<std::string> tokens;
  /** Per-token timestamps in seconds. */
  std::vector<float> timestamps;
  /** Segment start time in seconds. */
  float start_time = 0.0f;
  /** JSON representation of the result. */
  std::string json;
};

/** @brief Configuration for the C++ keyword spotting wrapper. */
struct KeywordSpotterConfig {
  /** Feature extraction configuration. */
  FeatureConfig feat_config;
  /** Streaming acoustic model configuration. */
  OnlineModelConfig model_config;
  /** Maximum number of active paths. */
  int32_t max_active_paths = 4;
  /** Number of trailing blanks required before finalizing a trigger. */
  int32_t num_trailing_blanks = 1;
  /** Keyword score bonus. */
  float keywords_score = 1.0f;
  /** Detection threshold. */
  float keywords_threshold = 0.25f;
  /** Keyword file. */
  std::string keywords_file;
  /** In-memory keyword definitions. */
  std::string keywords_buf;
};

/** @brief RAII wrapper for keyword spotting. */
class SHERPA_ONNX_API KeywordSpotter
    : public MoveOnly<KeywordSpotter, SherpaOnnxKeywordSpotter> {
 public:
  /** @brief Create a keyword spotter from a config struct. */
  static KeywordSpotter Create(const KeywordSpotterConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxKeywordSpotter *p) const;

  /** @brief Create a keyword stream using configured keywords. */
  OnlineStream CreateStream() const;

  /** @brief Create a keyword stream with inline extra or replacement keywords.
   */
  OnlineStream CreateStream(const std::string &keywords) const;

  /** @brief Check whether the stream has enough data to decode. */
  bool IsReady(const OnlineStream *s) const;

  /** @brief Decode one ready stream. */
  void Decode(const OnlineStream *s) const;

  /** @brief Decode multiple ready streams in parallel. */
  void Decode(const OnlineStream *ss, int32_t n) const;

  /** @brief Reset a stream after a keyword trigger. */
  void Reset(const OnlineStream *s) const;

  /** @brief Return the copied keyword spotting result for a stream. */
  KeywordResult GetResult(const OnlineStream *s) const;

 private:
  explicit KeywordSpotter(const SherpaOnnxKeywordSpotter *p);
};

/** @brief GTCRN speech denoiser model configuration. */
struct OfflineSpeechDenoiserGtcrnModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/** @brief DPDFNet speech denoiser model configuration. */
struct OfflineSpeechDenoiserDpdfNetModelConfig {
  /** Model ONNX file. */
  std::string model;
};

/**
 * @brief Speech denoiser model configuration.
 *
 * Configure exactly one model family. If multiple model families are set, one
 * is chosen and the choice is implementation-defined.
 */
struct OfflineSpeechDenoiserModelConfig {
  /** GTCRN configuration. */
  OfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  /** DPDFNet configuration. */
  OfflineSpeechDenoiserDpdfNetModelConfig dpdfnet;
  /** Number of inference threads. */
  int32_t num_threads = 1;
  /** Enable verbose debug logging. */
  bool debug = false;
  /** Execution provider such as `"cpu"`. */
  std::string provider = "cpu";
};

/** @brief Configuration for offline speech denoising. */
struct OfflineSpeechDenoiserConfig {
  /** Model configuration. */
  OfflineSpeechDenoiserModelConfig model;
};

/** @brief Denoised waveform returned by speech enhancement wrappers. */
struct DenoisedAudio {
  /** Output samples normalized to `[-1, 1]`. */
  std::vector<float> samples;
  /** Output sample rate in Hz. */
  int32_t sample_rate = 0;
};

/** @brief RAII wrapper for offline speech denoising. */
class SHERPA_ONNX_API OfflineSpeechDenoiser
    : public MoveOnly<OfflineSpeechDenoiser, SherpaOnnxOfflineSpeechDenoiser> {
 public:
  /** @brief Create an offline speech denoiser. */
  static OfflineSpeechDenoiser Create(
      const OfflineSpeechDenoiserConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOfflineSpeechDenoiser *p) const;

  /** @brief Run denoising on a complete waveform. */
  DenoisedAudio Run(const float *samples, int32_t n, int32_t sample_rate) const;

  /** @brief Return the expected input sample rate. */
  int32_t GetSampleRate() const;

 private:
  explicit OfflineSpeechDenoiser(const SherpaOnnxOfflineSpeechDenoiser *p);
};

/** @brief Configuration for online speech denoising. */
struct OnlineSpeechDenoiserConfig {
  /** Model configuration. */
  OfflineSpeechDenoiserModelConfig model;
};

/** @brief RAII wrapper for online speech denoising. */
class SHERPA_ONNX_API OnlineSpeechDenoiser
    : public MoveOnly<OnlineSpeechDenoiser, SherpaOnnxOnlineSpeechDenoiser> {
 public:
  /** @brief Create an online speech denoiser. */
  static OnlineSpeechDenoiser Create(const OnlineSpeechDenoiserConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOnlineSpeechDenoiser *p) const;

  /** @brief Process one chunk of streaming audio. */
  DenoisedAudio Run(const float *samples, int32_t n, int32_t sample_rate) const;

  /** @brief Flush buffered audio and reset the denoiser. */
  DenoisedAudio Flush() const;

  /** @brief Reset the denoiser for a new stream. */
  void Reset() const;

  /** @brief Return the expected input sample rate. */
  int32_t GetSampleRate() const;

  /** @brief Return the recommended frame shift in samples for streaming input.
   */
  int32_t GetFrameShiftInSamples() const;

 private:
  explicit OnlineSpeechDenoiser(const SherpaOnnxOnlineSpeechDenoiser *p);
};

// ==============================
// VAD
// ==============================

/** @brief Silero VAD model configuration. */
struct SileroVadModelConfig {
  /** Model ONNX file. */
  std::string model;
  /** Detection threshold. */
  float threshold = 0.5;
  /** Minimum silence duration in seconds. */
  float min_silence_duration = 0.5;
  /** Minimum speech duration in seconds. */
  float min_speech_duration = 0.25;
  /** Window size in samples. */
  int32_t window_size = 512;
  /** Maximum speech duration in seconds before forced split. */
  float max_speech_duration = 20;
};

/** @brief Ten VAD model configuration. */
struct TenVadModelConfig {
  /** Model ONNX file. */
  std::string model;
  /** Detection threshold. */
  float threshold = 0.5;
  /** Minimum silence duration in seconds. */
  float min_silence_duration = 0.5;
  /** Minimum speech duration in seconds. */
  float min_speech_duration = 0.25;
  /** Window size in samples. */
  int32_t window_size = 256;
  /** Maximum speech duration in seconds before forced split. */
  float max_speech_duration = 20;
};

/**
 * @brief VAD model configuration.
 *
 * Configure exactly one model family. If multiple model families are set, one
 * is chosen and the choice is implementation-defined.
 */
struct VadModelConfig {
  /** Silero VAD configuration. */
  SileroVadModelConfig silero_vad;
  /** Ten VAD configuration. */
  TenVadModelConfig ten_vad;

  /** Input sample rate in Hz. */
  int32_t sample_rate = 16000;
  /** Number of inference threads. */
  int32_t num_threads = 1;
  /** Execution provider such as `"cpu"`. */
  std::string provider = "cpu";
  /** Enable verbose debug logging. */
  bool debug = false;
};

/** @brief One speech segment produced by the VAD wrapper. */
struct SpeechSegment {
  /** Start sample index relative to the processed audio timeline. */
  int32_t start = 0;
  /** Speech samples for the segment. */
  std::vector<float> samples;
};

/** @brief RAII wrapper for the circular buffer helper used by VAD. */
class SHERPA_ONNX_API CircularBuffer
    : public MoveOnly<CircularBuffer, SherpaOnnxCircularBuffer> {
 public:
  /** @brief Create a circular buffer with the given capacity in samples. */
  static CircularBuffer Create(int32_t capacity);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxCircularBuffer *p) const;

  /** @brief Append samples to the buffer. */
  void Push(const float *p, int32_t n) const;

  /** @brief Copy a contiguous span from the buffer. */
  std::vector<float> Get(int32_t start_index, int32_t n) const;

  /** @brief Remove samples from the head of the buffer. */
  void Pop(int32_t n) const;

  /** @brief Return the number of stored samples. */
  int32_t Size() const;

  /** @brief Return the current head index. */
  int32_t Head() const;

  /** @brief Reset the buffer to empty. */
  void Reset() const;

 private:
  explicit CircularBuffer(const SherpaOnnxCircularBuffer *p);
};

/**
 * @brief RAII wrapper for voice activity detection.
 *
 * The wrapper collects detected speech segments internally. Use `IsEmpty()`,
 * `Front()`, and `Pop()` to consume them.
 */
class SHERPA_ONNX_API VoiceActivityDetector
    : public MoveOnly<VoiceActivityDetector, SherpaOnnxVoiceActivityDetector> {
 public:
  /** @brief Create a VAD instance. */
  static VoiceActivityDetector Create(const VadModelConfig &config,
                                      float buffer_size_in_seconds);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxVoiceActivityDetector *p) const;

  /** @brief Feed more audio samples to the detector. */
  void AcceptWaveform(const float *samples, int32_t n) const;

  /** @brief Check whether no speech segments are currently queued. */
  bool IsEmpty() const;

  /** @brief Check whether speech is currently detected. */
  bool IsDetected() const;

  /** @brief Remove the front queued speech segment. */
  void Pop() const;

  /** @brief Remove all queued speech segments. */
  void Clear() const;

  /** @brief Return the front queued speech segment. */
  SpeechSegment Front() const;

  /** @brief Like Front(), but returns the segment in a shared pointer. */
  std::shared_ptr<SpeechSegment> FrontPtr() const;

  /** @brief Reset the detector state. */
  void Reset() const;

  /** @brief Flush buffered context at end of input. */
  void Flush() const;

 private:
  explicit VoiceActivityDetector(const SherpaOnnxVoiceActivityDetector *p);
};

/** @brief RAII wrapper for linear resampling. */
class SHERPA_ONNX_API LinearResampler
    : public MoveOnly<LinearResampler, SherpaOnnxLinearResampler> {
 public:
  /** @brief Construct an empty wrapper. */
  LinearResampler() = default;
  /** @brief Create a linear resampler. */
  static LinearResampler Create(int32_t samp_rate_in_hz,
                                int32_t samp_rate_out_hz,
                                float filter_cutoff_hz, int32_t num_zeros);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxLinearResampler *p) const;

  /** @brief Reset the resampler state. */
  void Reset() const;

  /** @brief Resample one chunk of input audio. */
  std::vector<float> Resample(const float *input, int32_t input_dim,
                              bool flush) const;

  /** @brief Return the input sample rate in Hz. */
  int32_t GetInputSamplingRate() const;
  /** @brief Return the output sample rate in Hz. */
  int32_t GetOutputSamplingRate() const;

 private:
  explicit LinearResampler(const SherpaOnnxLinearResampler *p);
};

/** @brief Return the sherpa-onnx version string as a C++ string. */
SHERPA_ONNX_API std::string GetVersionStr();
/** @brief Return the build Git SHA1 as a C++ string. */
SHERPA_ONNX_API std::string GetGitSha1();
/** @brief Return the build Git date as a C++ string. */
SHERPA_ONNX_API std::string GetGitDate();
/** @brief Return `true` if a file exists. */
SHERPA_ONNX_API bool FileExists(const std::string &filename);

// ============================================================================
// Offline Punctuation
// ============================================================================
/** @brief Offline punctuation model configuration. */
struct OfflinePunctuationModelConfig {
  /** Model file. */
  std::string ct_transformer;
  /** Number of inference threads. */
  int32_t num_threads = 1;
  /** Enable verbose debug logging. */
  bool debug = false;
  /** Execution provider such as `"cpu"`. */
  std::string provider = "cpu";
};

/** @brief Configuration for offline punctuation. */
struct OfflinePunctuationConfig {
  /** Model configuration. */
  OfflinePunctuationModelConfig model;
};

/** @brief RAII wrapper for offline punctuation restoration. */
class SHERPA_ONNX_API OfflinePunctuation
    : public MoveOnly<OfflinePunctuation, SherpaOnnxOfflinePunctuation> {
 public:
  /** @brief Create an offline punctuation model. */
  static OfflinePunctuation Create(const OfflinePunctuationConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOfflinePunctuation *p) const;

  /** @brief Add punctuation to a complete input text. */
  std::string AddPunctuation(const std::string &text) const;

 private:
  explicit OfflinePunctuation(const SherpaOnnxOfflinePunctuation *p);
};

// ============================================================================
// Online Punctuation
// ============================================================================
/** @brief Online punctuation model configuration. */
struct OnlinePunctuationModelConfig {
  /** Model file. */
  std::string cnn_bilstm;
  /** BPE vocabulary file. */
  std::string bpe_vocab;
  /** Number of inference threads. */
  int32_t num_threads = 1;
  /** Enable verbose debug logging. */
  bool debug = false;
  /** Execution provider such as `"cpu"`. */
  std::string provider = "cpu";
};

/** @brief Configuration for online punctuation. */
struct OnlinePunctuationConfig {
  /** Model configuration. */
  OnlinePunctuationModelConfig model;
};

/** @brief RAII wrapper for online punctuation restoration. */
class SHERPA_ONNX_API OnlinePunctuation
    : public MoveOnly<OnlinePunctuation, SherpaOnnxOnlinePunctuation> {
 public:
  /** @brief Create an online punctuation model. */
  static OnlinePunctuation Create(const OnlinePunctuationConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOnlinePunctuation *p) const;

  /** @brief Add punctuation to one input text chunk. */
  std::string AddPunctuation(const std::string &text) const;

 private:
  explicit OnlinePunctuation(const SherpaOnnxOnlinePunctuation *p);
};

// ============================================================================
// Audio tagging
// ============================================================================
/** @brief Zipformer audio-tagging model configuration. */
struct OfflineZipformerAudioTaggingModelConfig {
  /** Model file. */
  std::string model;
};

/**
 * @brief Audio-tagging model configuration.
 *
 * Configure exactly one model family. If multiple model families are set, one
 * is chosen and the choice is implementation-defined.
 */
struct AudioTaggingModelConfig {
  /** Zipformer model configuration. */
  OfflineZipformerAudioTaggingModelConfig zipformer;
  /** Alternative CED model file. */
  std::string ced;
  /** Number of inference threads. */
  int32_t num_threads = 1;
  /** Enable verbose debug logging. */
  bool debug = false;
  /** Execution provider such as `"cpu"`. */
  std::string provider = "cpu";
};

/** @brief Configuration for audio tagging. */
struct AudioTaggingConfig {
  /** Model configuration. */
  AudioTaggingModelConfig model;
  /** CSV file containing label names. */
  std::string labels;
  /** Default number of results to return. */
  int32_t top_k = 5;
};

/** @brief One audio-tagging event returned by the C++ wrapper. */
struct AudioEvent {
  /** Event label. */
  std::string name;
  /** Class index. */
  int32_t index;
  /** Probability or confidence score. */
  float prob;
};

/** @brief RAII wrapper for audio tagging. */
class SHERPA_ONNX_API AudioTagging
    : public MoveOnly<AudioTagging, SherpaOnnxAudioTagging> {
 public:
  /** @brief Create an audio tagger. */
  static AudioTagging Create(const AudioTaggingConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxAudioTagging *p) const;

  /** @brief Create an offline stream for tagging. */
  OfflineStream CreateStream() const;
  /**
   * @brief Run audio tagging and return copied results.
   *
   * When `top_k == -1`, the wrapper uses `config.top_k`. When `top_k > 0`,
   * that argument overrides the configured default.
   */
  std::vector<AudioEvent> Compute(const OfflineStream *s, int32_t top_k = -1);

  /** @brief Like Compute(), but returns the result vector in a shared pointer.
   */
  std::shared_ptr<std::vector<AudioEvent>> ComputePtr(const OfflineStream *s,
                                                      int32_t top_k = -1);

 private:
  explicit AudioTagging(const SherpaOnnxAudioTagging *p);
};

// ==============================
// Source Separation
// ==============================

/** @brief Spleeter source-separation model configuration. */
struct OfflineSourceSeparationSpleeterModelConfig {
  /** Path to the vocals ONNX model. */
  std::string vocals;
  /** Path to the accompaniment ONNX model. */
  std::string accompaniment;
};

/** @brief UVR (MDX-Net) source-separation model configuration. */
struct OfflineSourceSeparationUvrModelConfig {
  /** Path to the UVR ONNX model. */
  std::string model;
};

/**
 * @brief Source-separation model configuration.
 *
 * Configure exactly one model family (Spleeter or UVR).
 */
struct OfflineSourceSeparationModelConfig {
  /** Spleeter configuration. */
  OfflineSourceSeparationSpleeterModelConfig spleeter;
  /** UVR configuration. */
  OfflineSourceSeparationUvrModelConfig uvr;
  /** Number of inference threads. */
  int32_t num_threads = 1;
  /** Enable verbose debug logging. */
  bool debug = false;
  /** Execution provider such as `"cpu"`. */
  std::string provider = "cpu";
};

/** @brief Configuration for offline source separation. */
struct OfflineSourceSeparationConfig {
  /** Model configuration. */
  OfflineSourceSeparationModelConfig model;
};

/** @brief A single stem (output track) with one or more channels. */
struct SourceSeparationStem {
  /** samples[c] contains the sample array for channel c. */
  std::vector<std::vector<float>> samples;
};

/** @brief Output of a source-separation run. */
struct SourceSeparationOutput {
  /** Separated stems. */
  std::vector<SourceSeparationStem> stems;
  /** Sample rate in Hz. */
  int32_t sample_rate = 0;
};

/** @brief RAII wrapper for offline source separation. */
class SHERPA_ONNX_API OfflineSourceSeparation
    : public MoveOnly<OfflineSourceSeparation,
                      SherpaOnnxOfflineSourceSeparation> {
 public:
  /** @brief Create an offline source separation engine. */
  static OfflineSourceSeparation Create(
      const OfflineSourceSeparationConfig &config);

  /** @brief Destroy the wrapped C handle. */
  void Destroy(const SherpaOnnxOfflineSourceSeparation *p) const;

  /**
   * @brief Run source separation on multi-channel audio.
   *
   * @param samples      samples[c] is a float array for channel c.
   * @param num_channels Number of input channels.
   * @param num_samples  Number of samples per channel.
   * @param sample_rate  Input sample rate in Hz.
   * @return Separated stems, or an empty output on error.
   */
  SourceSeparationOutput Process(const float *const *samples,
                                 int32_t num_channels, int32_t num_samples,
                                 int32_t sample_rate) const;

  /** @brief Return the output sample rate. */
  int32_t GetOutputSampleRate() const;

  /** @brief Return the number of stems produced. */
  int32_t GetNumberOfStems() const;

 private:
  explicit OfflineSourceSeparation(const SherpaOnnxOfflineSourceSeparation *p);
};

}  // namespace sherpa_onnx::cxx

#endif  // SHERPA_ONNX_C_API_CXX_API_H_
