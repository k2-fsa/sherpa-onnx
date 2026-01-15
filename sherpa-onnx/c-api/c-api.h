// sherpa-onnx/c-api/c-api.h
//
// Copyright (c)  2023  Xiaomi Corporation

// C API for sherpa-onnx
//
// Please refer to
// https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/decode-file-c-api.c
// for usages.
//

#ifndef SHERPA_ONNX_C_API_C_API_H_
#define SHERPA_ONNX_C_API_C_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// See https://github.com/pytorch/pytorch/blob/main/c10/macros/Export.h
// We will set SHERPA_ONNX_BUILD_SHARED_LIBS and SHERPA_ONNX_BUILD_MAIN_LIB in
// CMakeLists.txt

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#if defined(_WIN32)
#if defined(SHERPA_ONNX_BUILD_SHARED_LIBS)
#define SHERPA_ONNX_EXPORT __declspec(dllexport)
#define SHERPA_ONNX_IMPORT __declspec(dllimport)
#else
#define SHERPA_ONNX_EXPORT
#define SHERPA_ONNX_IMPORT
#endif
#else  // WIN32
#define SHERPA_ONNX_EXPORT __attribute__((visibility("default")))

#define SHERPA_ONNX_IMPORT SHERPA_ONNX_EXPORT
#endif  // WIN32

#if defined(SHERPA_ONNX_BUILD_MAIN_LIB)
#define SHERPA_ONNX_API SHERPA_ONNX_EXPORT
#else
#define SHERPA_ONNX_API SHERPA_ONNX_IMPORT
#endif

// Please don't free the returned pointer.
// Please don't modify the memory pointed by the returned pointer.
//
// The memory pointed by the returned pointer is statically allocated.
//
// Example return value: "1.12.1"
SHERPA_ONNX_API const char *SherpaOnnxGetVersionStr();

// Please don't free the returned pointer.
// Please don't modify the memory pointed by the returned pointer.
//
// The memory pointed by the returned pointer is statically allocated.
//
// Example return value: "6982b86c"
SHERPA_ONNX_API const char *SherpaOnnxGetGitSha1();

// Please don't free the returned pointer.
// Please don't modify the memory pointed by the returned pointer.
//
// The memory pointed by the returned pointer is statically allocated.
//
// Example return value: "Fri Jun 20 11:22:52 2025"
SHERPA_ONNX_API const char *SherpaOnnxGetGitDate();

// return 1 if the given file exists; return 0 otherwise
SHERPA_ONNX_API int32_t SherpaOnnxFileExists(const char *filename);

/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to download pre-trained models. That is, you can find encoder-xxx.onnx
/// decoder-xxx.onnx, joiner-xxx.onnx, and tokens.txt for this struct
/// from there.
SHERPA_ONNX_API typedef struct SherpaOnnxOnlineTransducerModelConfig {
  const char *encoder;
  const char *decoder;
  const char *joiner;
} SherpaOnnxOnlineTransducerModelConfig;

// please visit
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/index.html
// to download pre-trained streaming paraformer models
SHERPA_ONNX_API typedef struct SherpaOnnxOnlineParaformerModelConfig {
  const char *encoder;
  const char *decoder;
} SherpaOnnxOnlineParaformerModelConfig;

// Please visit
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/zipformer-ctc-models.html#
// to download pre-trained streaming zipformer2 ctc models
SHERPA_ONNX_API typedef struct SherpaOnnxOnlineZipformer2CtcModelConfig {
  const char *model;
} SherpaOnnxOnlineZipformer2CtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOnlineNemoCtcModelConfig {
  const char *model;
} SherpaOnnxOnlineNemoCtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOnlineToneCtcModelConfig {
  const char *model;
} SherpaOnnxOnlineToneCtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOnlineModelConfig {
  SherpaOnnxOnlineTransducerModelConfig transducer;
  SherpaOnnxOnlineParaformerModelConfig paraformer;
  SherpaOnnxOnlineZipformer2CtcModelConfig zipformer2_ctc;
  const char *tokens;
  int32_t num_threads;
  const char *provider;
  int32_t debug;  // true to print debug information of the model
  const char *model_type;
  // Valid values:
  //  - cjkchar
  //  - bpe
  //  - cjkchar+bpe
  const char *modeling_unit;
  const char *bpe_vocab;
  /// if non-null, loading the tokens from the buffer instead of from the
  /// "tokens" file
  const char *tokens_buf;
  /// byte size excluding the trailing '\0'
  int32_t tokens_buf_size;
  SherpaOnnxOnlineNemoCtcModelConfig nemo_ctc;
  SherpaOnnxOnlineToneCtcModelConfig t_one_ctc;
} SherpaOnnxOnlineModelConfig;

/// It expects 16 kHz 16-bit single channel wave format.
SHERPA_ONNX_API typedef struct SherpaOnnxFeatureConfig {
  /// Sample rate of the input data. MUST match the one expected
  /// by the model. For instance, it should be 16000 for models provided
  /// by us.
  int32_t sample_rate;

  /// Feature dimension of the model.
  /// For instance, it should be 80 for models provided by us.
  int32_t feature_dim;
} SherpaOnnxFeatureConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOnlineCtcFstDecoderConfig {
  const char *graph;
  int32_t max_active;
} SherpaOnnxOnlineCtcFstDecoderConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxHomophoneReplacerConfig {
  const char *dict_dir;  // unused
  const char *lexicon;
  const char *rule_fsts;
} SherpaOnnxHomophoneReplacerConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOnlineRecognizerConfig {
  SherpaOnnxFeatureConfig feat_config;
  SherpaOnnxOnlineModelConfig model_config;

  /// Possible values are: greedy_search, modified_beam_search
  const char *decoding_method;

  /// Used only when decoding_method is modified_beam_search
  /// Example value: 4
  int32_t max_active_paths;

  /// 0 to disable endpoint detection.
  /// A non-zero value to enable endpoint detection.
  int32_t enable_endpoint;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value even if nothing has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule1_min_trailing_silence;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value after something that is not blank has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule2_min_trailing_silence;

  /// An endpoint is detected if the utterance in seconds is larger than
  /// this value.
  /// Used only when enable_endpoint is not 0.
  float rule3_min_utterance_length;

  /// Path to the hotwords.
  const char *hotwords_file;

  /// Bonus score for each token in hotwords.
  float hotwords_score;

  SherpaOnnxOnlineCtcFstDecoderConfig ctc_fst_decoder_config;
  const char *rule_fsts;
  const char *rule_fars;
  float blank_penalty;

  /// if non-nullptr, loading the hotwords from the buffered string directly in
  const char *hotwords_buf;
  /// byte size excluding the tailing '\0'
  int32_t hotwords_buf_size;
  SherpaOnnxHomophoneReplacerConfig hr;
} SherpaOnnxOnlineRecognizerConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOnlineRecognizerResult {
  // Recognized text
  const char *text;

  // Pointer to continuous memory which holds string based tokens
  // which are separated by \0
  const char *tokens;

  // a pointer array containing the address of the first item in tokens
  const char *const *tokens_arr;

  // Pointer to continuous memory which holds timestamps
  //
  // Caution: If timestamp information is not available, this pointer is NULL.
  // Please check whether it is NULL before you access it; otherwise, you would
  // get segmentation fault.
  float *timestamps;

  // The number of tokens/timestamps in above pointer
  int32_t count;

  /** Return a json string.
   *
   * The returned string contains:
   *   {
   *     "text": "The recognition result",
   *     "tokens": [x, x, x],
   *     "timestamps": [x, x, x],
   *     "segment": x,
   *     "start_time": x,
   *     "is_final": true|false
   *   }
   */
  const char *json;
} SherpaOnnxOnlineRecognizerResult;

/// Note: OnlineRecognizer here means StreamingRecognizer.
/// It does not need to access the Internet during recognition.
/// Everything is run locally.
SHERPA_ONNX_API typedef struct SherpaOnnxOnlineRecognizer
    SherpaOnnxOnlineRecognizer;
SHERPA_ONNX_API typedef struct SherpaOnnxOnlineStream SherpaOnnxOnlineStream;

/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          SherpaOnnxDestroyOnlineRecognizer() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOnlineRecognizer *
SherpaOnnxCreateOnlineRecognizer(
    const SherpaOnnxOnlineRecognizerConfig *config);

/// Free a pointer returned by SherpaOnnxCreateOnlineRecognizer()
///
/// @param p A pointer returned by SherpaOnnxCreateOnlineRecognizer()
SHERPA_ONNX_API void SherpaOnnxDestroyOnlineRecognizer(
    const SherpaOnnxOnlineRecognizer *recognizer);

/// Create an online stream for accepting wave samples.
///
/// @param recognizer  A pointer returned by SherpaOnnxCreateOnlineRecognizer()
/// @return Return a pointer to an OnlineStream. The user has to invoke
///         SherpaOnnxDestroyOnlineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOnlineStream *SherpaOnnxCreateOnlineStream(
    const SherpaOnnxOnlineRecognizer *recognizer);

/// Create an online stream for accepting wave samples with the specified hot
/// words.
///
/// @param recognizer  A pointer returned by SherpaOnnxCreateOnlineRecognizer()
/// @return Return a pointer to an OnlineStream. The user has to invoke
///         SherpaOnnxDestroyOnlineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOnlineStream *
SherpaOnnxCreateOnlineStreamWithHotwords(
    const SherpaOnnxOnlineRecognizer *recognizer, const char *hotwords);

/// Destroy an online stream.
///
/// @param stream A pointer returned by SherpaOnnxCreateOnlineStream()
SHERPA_ONNX_API void SherpaOnnxDestroyOnlineStream(
    const SherpaOnnxOnlineStream *stream);

/// Accept input audio samples and compute the features.
/// The user has to invoke SherpaOnnxDecodeOnlineStream() to run the neural
/// network and decoding.
///
/// @param stream  A pointer returned by SherpaOnnxCreateOnlineStream().
/// @param sample_rate  Sample rate of the input samples. If it is different
///                     from config.feat_config.sample_rate, we will do
///                     resampling inside sherpa-onnx.
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
SHERPA_ONNX_API void SherpaOnnxOnlineStreamAcceptWaveform(
    const SherpaOnnxOnlineStream *stream, int32_t sample_rate,
    const float *samples, int32_t n);

/// Return 1 if there are enough number of feature frames for decoding.
/// Return 0 otherwise.
///
/// @param recognizer  A pointer returned by SherpaOnnxCreateOnlineRecognizer
/// @param stream  A pointer returned by SherpaOnnxCreateOnlineStream
SHERPA_ONNX_API int32_t
SherpaOnnxIsOnlineStreamReady(const SherpaOnnxOnlineRecognizer *recognizer,
                              const SherpaOnnxOnlineStream *stream);

/// Call this function to run the neural network model and decoding.
//
/// Precondition for this function: SherpaOnnxIsOnlineStreamReady() MUST
/// return 1.
///
/// Usage example:
///
///  while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
///     SherpaOnnxDecodeOnlineStream(recognizer, stream);
///  }
///
SHERPA_ONNX_API void SherpaOnnxDecodeOnlineStream(
    const SherpaOnnxOnlineRecognizer *recognizer,
    const SherpaOnnxOnlineStream *stream);

/// This function is similar to SherpaOnnxDecodeOnlineStream(). It decodes
/// multiple OnlineStream in parallel.
///
/// Caution: The caller has to ensure each OnlineStream is ready, i.e.,
/// SherpaOnnxIsOnlineStreamReady() for that stream should return 1.
///
/// @param recognizer  A pointer returned by SherpaOnnxCreateOnlineRecognizer()
/// @param streams  A pointer array containing pointers returned by
///                 SherpaOnnxCreateOnlineRecognizer()
/// @param n  Number of elements in the given streams array.
SHERPA_ONNX_API void SherpaOnnxDecodeMultipleOnlineStreams(
    const SherpaOnnxOnlineRecognizer *recognizer,
    const SherpaOnnxOnlineStream **streams, int32_t n);

/// Get the decoding results so far for an OnlineStream.
///
/// @param recognizer A pointer returned by SherpaOnnxCreateOnlineRecognizer().
/// @param stream A pointer returned by SherpaOnnxCreateOnlineStream().
/// @return A pointer containing the result. The user has to invoke
///         SherpaOnnxDestroyOnlineRecognizerResult() to free the returned
///         pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOnlineRecognizerResult *
SherpaOnnxGetOnlineStreamResult(const SherpaOnnxOnlineRecognizer *recognizer,
                                const SherpaOnnxOnlineStream *stream);

/// Destroy the pointer returned by SherpaOnnxGetOnlineStreamResult().
///
/// @param r A pointer returned by SherpaOnnxGetOnlineStreamResult()
SHERPA_ONNX_API void SherpaOnnxDestroyOnlineRecognizerResult(
    const SherpaOnnxOnlineRecognizerResult *r);

/// Return the result as a json string.
/// The user has to invoke
/// SherpaOnnxDestroyOnlineStreamResultJson()
/// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const char *SherpaOnnxGetOnlineStreamResultAsJson(
    const SherpaOnnxOnlineRecognizer *recognizer,
    const SherpaOnnxOnlineStream *stream);

SHERPA_ONNX_API void SherpaOnnxDestroyOnlineStreamResultJson(const char *s);

/// SherpaOnnxOnlineStreamReset an OnlineStream , which clears the neural
/// network model state and the state for decoding.
///
/// @param recognizer A pointer returned by SherpaOnnxCreateOnlineRecognizer().
/// @param stream A pointer returned by SherpaOnnxCreateOnlineStream
SHERPA_ONNX_API void SherpaOnnxOnlineStreamReset(
    const SherpaOnnxOnlineRecognizer *recognizer,
    const SherpaOnnxOnlineStream *stream);

/// Signal that no more audio samples would be available.
/// After this call, you cannot call SherpaOnnxOnlineStreamAcceptWaveform() any
/// more.
///
/// @param stream A pointer returned by SherpaOnnxCreateOnlineStream()
SHERPA_ONNX_API void SherpaOnnxOnlineStreamInputFinished(
    const SherpaOnnxOnlineStream *stream);

/// Return 1 if an endpoint has been detected.
///
/// @param recognizer A pointer returned by SherpaOnnxCreateOnlineRecognizer()
/// @param stream A pointer returned by SherpaOnnxCreateOnlineStream()
/// @return Return 1 if an endpoint is detected. Return 0 otherwise.
SHERPA_ONNX_API int32_t
SherpaOnnxOnlineStreamIsEndpoint(const SherpaOnnxOnlineRecognizer *recognizer,
                                 const SherpaOnnxOnlineStream *stream);

// for displaying results on Linux/macOS.
SHERPA_ONNX_API typedef struct SherpaOnnxDisplay SherpaOnnxDisplay;

/// Create a display object. Must be freed using SherpaOnnxDestroyDisplay to
/// avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxDisplay *SherpaOnnxCreateDisplay(
    int32_t max_word_per_line);

SHERPA_ONNX_API void SherpaOnnxDestroyDisplay(const SherpaOnnxDisplay *display);

/// Print the result.
SHERPA_ONNX_API void SherpaOnnxPrint(const SherpaOnnxDisplay *display,
                                     int32_t idx, const char *s);
// ============================================================
// For offline ASR (i.e., non-streaming ASR)
// ============================================================

/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to download pre-trained models. That is, you can find encoder-xxx.onnx
/// decoder-xxx.onnx, and joiner-xxx.onnx for this struct
/// from there.
SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTransducerModelConfig {
  const char *encoder;
  const char *decoder;
  const char *joiner;
} SherpaOnnxOfflineTransducerModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineParaformerModelConfig {
  const char *model;
} SherpaOnnxOfflineParaformerModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineNemoEncDecCtcModelConfig {
  const char *model;
} SherpaOnnxOfflineNemoEncDecCtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineWhisperModelConfig {
  const char *encoder;
  const char *decoder;
  const char *language;
  const char *task;
  int32_t tail_paddings;
} SherpaOnnxOfflineWhisperModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineCanaryModelConfig {
  const char *encoder;
  const char *decoder;
  const char *src_lang;
  const char *tgt_lang;
  int32_t use_pnc;
} SherpaOnnxOfflineCanaryModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineFireRedAsrModelConfig {
  const char *encoder;
  const char *decoder;
} SherpaOnnxOfflineFireRedAsrModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineMoonshineModelConfig {
  const char *preprocessor;
  const char *encoder;
  const char *uncached_decoder;
  const char *cached_decoder;
} SherpaOnnxOfflineMoonshineModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTdnnModelConfig {
  const char *model;
} SherpaOnnxOfflineTdnnModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineLMConfig {
  const char *model;
  float scale;
} SherpaOnnxOfflineLMConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSenseVoiceModelConfig {
  const char *model;
  const char *language;
  int32_t use_itn;
} SherpaOnnxOfflineSenseVoiceModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineDolphinModelConfig {
  const char *model;
} SherpaOnnxOfflineDolphinModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineZipformerCtcModelConfig {
  const char *model;
} SherpaOnnxOfflineZipformerCtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineWenetCtcModelConfig {
  const char *model;
} SherpaOnnxOfflineWenetCtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineOmnilingualAsrCtcModelConfig {
  const char *model;
} SherpaOnnxOfflineOmnilingualAsrCtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineFunASRNanoModelConfig {
  const char *encoder_adaptor;
  const char *llm;
  const char *embedding;
  const char *tokenizer;
  const char *system_prompt;
  const char *user_prompt;
  int32_t max_new_tokens;
  float temperature;
  float top_p;
  int32_t seed;
} SherpaOnnxOfflineFunASRNanoModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineMedAsrCtcModelConfig {
  const char *model;
} SherpaOnnxOfflineMedAsrCtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineModelConfig {
  SherpaOnnxOfflineTransducerModelConfig transducer;
  SherpaOnnxOfflineParaformerModelConfig paraformer;
  SherpaOnnxOfflineNemoEncDecCtcModelConfig nemo_ctc;
  SherpaOnnxOfflineWhisperModelConfig whisper;
  SherpaOnnxOfflineTdnnModelConfig tdnn;

  const char *tokens;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
  const char *model_type;
  // Valid values:
  //  - cjkchar
  //  - bpe
  //  - cjkchar+bpe
  const char *modeling_unit;
  const char *bpe_vocab;
  const char *telespeech_ctc;
  SherpaOnnxOfflineSenseVoiceModelConfig sense_voice;
  SherpaOnnxOfflineMoonshineModelConfig moonshine;
  SherpaOnnxOfflineFireRedAsrModelConfig fire_red_asr;
  SherpaOnnxOfflineDolphinModelConfig dolphin;
  SherpaOnnxOfflineZipformerCtcModelConfig zipformer_ctc;
  SherpaOnnxOfflineCanaryModelConfig canary;
  SherpaOnnxOfflineWenetCtcModelConfig wenet_ctc;
  SherpaOnnxOfflineOmnilingualAsrCtcModelConfig omnilingual;
  SherpaOnnxOfflineMedAsrCtcModelConfig medasr;
  SherpaOnnxOfflineFunASRNanoModelConfig funasr_nano;
} SherpaOnnxOfflineModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineRecognizerConfig {
  SherpaOnnxFeatureConfig feat_config;
  SherpaOnnxOfflineModelConfig model_config;
  SherpaOnnxOfflineLMConfig lm_config;

  const char *decoding_method;
  int32_t max_active_paths;

  /// Path to the hotwords.
  const char *hotwords_file;

  /// Bonus score for each token in hotwords.
  float hotwords_score;
  const char *rule_fsts;
  const char *rule_fars;
  float blank_penalty;

  SherpaOnnxHomophoneReplacerConfig hr;
} SherpaOnnxOfflineRecognizerConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineRecognizer
    SherpaOnnxOfflineRecognizer;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineStream SherpaOnnxOfflineStream;

/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          SherpaOnnxDestroyOfflineRecognizer() to free it to avoid memory
//          leak.
SHERPA_ONNX_API const SherpaOnnxOfflineRecognizer *
SherpaOnnxCreateOfflineRecognizer(
    const SherpaOnnxOfflineRecognizerConfig *config);

/// @param config  Config for the recognizer.
SHERPA_ONNX_API void SherpaOnnxOfflineRecognizerSetConfig(
    const SherpaOnnxOfflineRecognizer *recognizer,
    const SherpaOnnxOfflineRecognizerConfig *config);

/// Free a pointer returned by SherpaOnnxCreateOfflineRecognizer()
///
/// @param p A pointer returned by SherpaOnnxCreateOfflineRecognizer()
SHERPA_ONNX_API void SherpaOnnxDestroyOfflineRecognizer(
    const SherpaOnnxOfflineRecognizer *recognizer);

/// Create an offline stream for accepting wave samples.
///
/// @param recognizer  A pointer returned by SherpaOnnxCreateOfflineRecognizer()
/// @return Return a pointer to an OfflineStream. The user has to invoke
///         SherpaOnnxDestroyOfflineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOfflineStream *SherpaOnnxCreateOfflineStream(
    const SherpaOnnxOfflineRecognizer *recognizer);

/// Create an offline stream for accepting wave samples with the specified hot
/// words.
///
/// @param recognizer  A pointer returned by SherpaOnnxCreateOfflineRecognizer()
/// @return Return a pointer to an OfflineStream. The user has to invoke
///         SherpaOnnxDestroyOfflineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOfflineStream *
SherpaOnnxCreateOfflineStreamWithHotwords(
    const SherpaOnnxOfflineRecognizer *recognizer, const char *hotwords);

/// Destroy an offline stream.
///
/// @param stream A pointer returned by SherpaOnnxCreateOfflineStream()
SHERPA_ONNX_API void SherpaOnnxDestroyOfflineStream(
    const SherpaOnnxOfflineStream *stream);

/// Accept input audio samples and compute the features.
/// The user has to invoke SherpaOnnxDecodeOfflineStream() to run the neural
/// network and decoding.
///
/// @param stream  A pointer returned by SherpaOnnxCreateOfflineStream().
/// @param sample_rate  Sample rate of the input samples. If it is different
///                     from config.feat_config.sample_rate, we will do
///                     resampling inside sherpa-onnx.
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
///
/// @caution: For each offline stream, please invoke this function only once!
SHERPA_ONNX_API void SherpaOnnxAcceptWaveformOffline(
    const SherpaOnnxOfflineStream *stream, int32_t sample_rate,
    const float *samples, int32_t n);
/// Decode an offline stream.
///
/// We assume you have invoked SherpaOnnxAcceptWaveformOffline() for the given
/// stream before calling this function.
///
/// @param recognizer A pointer returned by SherpaOnnxCreateOfflineRecognizer().
/// @param stream A pointer returned by SherpaOnnxCreateOfflineStream()
SHERPA_ONNX_API void SherpaOnnxDecodeOfflineStream(
    const SherpaOnnxOfflineRecognizer *recognizer,
    const SherpaOnnxOfflineStream *stream);

/// Decode a list offline streams in parallel.
///
/// We assume you have invoked SherpaOnnxAcceptWaveformOffline() for each stream
/// before calling this function.
///
/// @param recognizer A pointer returned by SherpaOnnxCreateOfflineRecognizer().
/// @param streams A pointer pointer array containing pointers returned
///                by SherpaOnnxCreateOfflineStream().
/// @param n Number of entries in the given streams.
SHERPA_ONNX_API void SherpaOnnxDecodeMultipleOfflineStreams(
    const SherpaOnnxOfflineRecognizer *recognizer,
    const SherpaOnnxOfflineStream **streams, int32_t n);

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineRecognizerResult {
  const char *text;

  // Pointer to continuous memory which holds timestamps
  //
  // It is NULL if the model does not support timestamps
  float *timestamps;

  // number of entries in timestamps
  int32_t count;

  // Pointer to continuous memory which holds string based tokens
  // which are separated by \0
  const char *tokens;

  // a pointer array containing the address of the first item in tokens
  const char *const *tokens_arr;

  /** Return a json string.
   *
   * The returned string contains:
   *   {
   *     "text": "The recognition result",
   *     "tokens": [x, x, x],
   *     "timestamps": [x, x, x],
   *     "durations": [x, x, x],
   *     "segment": x,
   *     "start_time": x,
   *     "is_final": true|false
   *   }
   */
  const char *json;

  // return recognized language
  const char *lang;

  // return emotion.
  const char *emotion;

  // return event.
  const char *event;

  // Pointer to continuous memory which holds durations (in seconds) for each
  // token It is NULL if the model does not support durations
  float *durations;

  // Pointer to continuous memory which holds log probabilities (confidence)
  // for each token. It is NULL if the model does not support probabilities.
  // ys_log_probs[i] is the log probability for token i.
  float *ys_log_probs;
} SherpaOnnxOfflineRecognizerResult;

/// Get the result of the offline stream.
///
/// We assume you have called SherpaOnnxDecodeOfflineStream() or
/// SherpaOnnxDecodeMultipleOfflineStreams() with the given stream before
/// calling this function.
///
/// @param stream A pointer returned by SherpaOnnxCreateOfflineStream().
/// @return Return a pointer to the result. The user has to invoke
///         SherpaOnnxDestroyOnlineRecognizerResult() to free the returned
///         pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOfflineRecognizerResult *
SherpaOnnxGetOfflineStreamResult(const SherpaOnnxOfflineStream *stream);

/// Destroy the pointer returned by SherpaOnnxGetOfflineStreamResult().
///
/// @param r A pointer returned by SherpaOnnxGetOfflineStreamResult()
SHERPA_ONNX_API void SherpaOnnxDestroyOfflineRecognizerResult(
    const SherpaOnnxOfflineRecognizerResult *r);

/// Return the result as a json string.
/// The user has to use SherpaOnnxDestroyOfflineStreamResultJson()
/// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const char *SherpaOnnxGetOfflineStreamResultAsJson(
    const SherpaOnnxOfflineStream *stream);

SHERPA_ONNX_API void SherpaOnnxDestroyOfflineStreamResultJson(const char *s);

// ============================================================
// For Keyword Spotter
// ============================================================
SHERPA_ONNX_API typedef struct SherpaOnnxKeywordResult {
  /// The triggered keyword.
  /// For English, it consists of space separated words.
  /// For Chinese, it consists of Chinese words without spaces.
  /// Example 1: "hello world"
  /// Example 2: "你好世界"
  const char *keyword;

  /// Decoded results at the token level.
  /// For instance, for BPE-based models it consists of a list of BPE tokens.
  const char *tokens;

  const char *const *tokens_arr;

  int32_t count;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  float *timestamps;

  /// Starting time of this segment.
  /// When an endpoint is detected, it will change
  float start_time;

  /** Return a json string.
   *
   * The returned string contains:
   *   {
   *     "keyword": "The triggered keyword",
   *     "tokens": [x, x, x],
   *     "timestamps": [x, x, x],
   *     "start_time": x,
   *   }
   */
  const char *json;
} SherpaOnnxKeywordResult;

SHERPA_ONNX_API typedef struct SherpaOnnxKeywordSpotterConfig {
  SherpaOnnxFeatureConfig feat_config;
  SherpaOnnxOnlineModelConfig model_config;
  int32_t max_active_paths;
  int32_t num_trailing_blanks;
  float keywords_score;
  float keywords_threshold;
  const char *keywords_file;
  /// if non-null, loading the keywords from the buffer instead of from the
  /// keywords_file
  const char *keywords_buf;
  /// byte size excluding the trailing '\0'
  int32_t keywords_buf_size;
} SherpaOnnxKeywordSpotterConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxKeywordSpotter
    SherpaOnnxKeywordSpotter;

/// @param config  Config for the keyword spotter.
/// @return Return a pointer to the spotter. The user has to invoke
///         SherpaOnnxDestroyKeywordSpotter() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxKeywordSpotter *SherpaOnnxCreateKeywordSpotter(
    const SherpaOnnxKeywordSpotterConfig *config);

/// Free a pointer returned by SherpaOnnxCreateKeywordSpotter()
///
/// @param p A pointer returned by SherpaOnnxCreateKeywordSpotter()
SHERPA_ONNX_API void SherpaOnnxDestroyKeywordSpotter(
    const SherpaOnnxKeywordSpotter *spotter);

/// Create an online stream for accepting wave samples.
///
/// @param spotter A pointer returned by SherpaOnnxCreateKeywordSpotter()
/// @return Return a pointer to an OnlineStream. The user has to invoke
///         SherpaOnnxDestroyOnlineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOnlineStream *SherpaOnnxCreateKeywordStream(
    const SherpaOnnxKeywordSpotter *spotter);

/// Create an online stream for accepting wave samples with the specified hot
/// words.
///
/// @param spotter A pointer returned by SherpaOnnxCreateKeywordSpotter()
/// @param keywords A pointer points to the keywords that you set
/// @return Return a pointer to an OnlineStream. The user has to invoke
///         SherpaOnnxDestroyOnlineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOnlineStream *
SherpaOnnxCreateKeywordStreamWithKeywords(
    const SherpaOnnxKeywordSpotter *spotter, const char *keywords);

/// Return 1 if there are enough number of feature frames for decoding.
/// Return 0 otherwise.
///
/// @param spotter A pointer returned by SherpaOnnxCreateKeywordSpotter
/// @param stream  A pointer returned by SherpaOnnxCreateKeywordStream
SHERPA_ONNX_API int32_t
SherpaOnnxIsKeywordStreamReady(const SherpaOnnxKeywordSpotter *spotter,
                               const SherpaOnnxOnlineStream *stream);

/// Call this function to run the neural network model and decoding.
//
/// Precondition for this function: SherpaOnnxIsKeywordStreamReady() MUST
/// return 1.
SHERPA_ONNX_API void SherpaOnnxDecodeKeywordStream(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream *stream);

/// Please call it right after a keyword is detected
SHERPA_ONNX_API void SherpaOnnxResetKeywordStream(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream *stream);

/// This function is similar to SherpaOnnxDecodeKeywordStream(). It decodes
/// multiple OnlineStream in parallel.
///
/// Caution: The caller has to ensure each OnlineStream is ready, i.e.,
/// SherpaOnnxIsKeywordStreamReady() for that stream should return 1.
///
/// @param spotter A pointer returned by SherpaOnnxCreateKeywordSpotter()
/// @param streams  A pointer array containing pointers returned by
///                 SherpaOnnxCreateKeywordStream()
/// @param n  Number of elements in the given streams array.
SHERPA_ONNX_API void SherpaOnnxDecodeMultipleKeywordStreams(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream **streams, int32_t n);

/// Get the decoding results so far for an OnlineStream.
///
/// @param spotter A pointer returned by SherpaOnnxCreateKeywordSpotter().
/// @param stream A pointer returned by SherpaOnnxCreateKeywordStream().
/// @return A pointer containing the result. The user has to invoke
///         SherpaOnnxDestroyKeywordResult() to free the returned pointer to
///         avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxKeywordResult *SherpaOnnxGetKeywordResult(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream *stream);

/// Destroy the pointer returned by SherpaOnnxGetKeywordResult().
///
/// @param r A pointer returned by SherpaOnnxGetKeywordResult()
SHERPA_ONNX_API void SherpaOnnxDestroyKeywordResult(
    const SherpaOnnxKeywordResult *r);

// the user has to call SherpaOnnxFreeKeywordResultJson() to free the returned
// pointer to avoid memory leak
SHERPA_ONNX_API const char *SherpaOnnxGetKeywordResultAsJson(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream *stream);

SHERPA_ONNX_API void SherpaOnnxFreeKeywordResultJson(const char *s);

// ============================================================
// For VAD
// ============================================================

SHERPA_ONNX_API typedef struct SherpaOnnxSileroVadModelConfig {
  // Path to the silero VAD model
  const char *model;

  // threshold to classify a segment as speech
  //
  // If the predicted probability of a segment is larger than this
  // value, then it is classified as speech.
  float threshold;

  // in seconds
  float min_silence_duration;

  // in seconds
  float min_speech_duration;

  int32_t window_size;

  // If a speech segment is longer than this value, then we increase
  // the threshold to 0.9. After finishing detecting the segment,
  // the threshold value is reset to its original value.
  float max_speech_duration;
} SherpaOnnxSileroVadModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxTenVadModelConfig {
  // Path to the ten-vad model
  const char *model;

  // threshold to classify a segment as speech
  //
  // If the predicted probability of a segment is larger than this
  // value, then it is classified as speech.
  float threshold;

  // in seconds
  float min_silence_duration;

  // in seconds
  float min_speech_duration;

  int32_t window_size;

  // If a speech segment is longer than this value, then we increase
  // the threshold to 0.9. After finishing detecting the segment,
  // the threshold value is reset to its original value.
  float max_speech_duration;
} SherpaOnnxTenVadModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxVadModelConfig {
  SherpaOnnxSileroVadModelConfig silero_vad;

  int32_t sample_rate;
  int32_t num_threads;
  const char *provider;
  int32_t debug;
  SherpaOnnxTenVadModelConfig ten_vad;
} SherpaOnnxVadModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxCircularBuffer
    SherpaOnnxCircularBuffer;

// Return an instance of circular buffer. The user has to use
// SherpaOnnxDestroyCircularBuffer() to free the returned pointer to avoid
// memory leak.
SHERPA_ONNX_API const SherpaOnnxCircularBuffer *SherpaOnnxCreateCircularBuffer(
    int32_t capacity);

// Free the pointer returned by SherpaOnnxCreateCircularBuffer()
SHERPA_ONNX_API void SherpaOnnxDestroyCircularBuffer(
    const SherpaOnnxCircularBuffer *buffer);

SHERPA_ONNX_API void SherpaOnnxCircularBufferPush(
    const SherpaOnnxCircularBuffer *buffer, const float *p, int32_t n);

// Return n samples starting at the given index.
//
// Return a pointer to an array containing n samples starting at start_index.
// The user has to use SherpaOnnxCircularBufferFree() to free the returned
// pointer to avoid memory leak.
SHERPA_ONNX_API const float *SherpaOnnxCircularBufferGet(
    const SherpaOnnxCircularBuffer *buffer, int32_t start_index, int32_t n);

// Free the pointer returned by SherpaOnnxCircularBufferGet().
SHERPA_ONNX_API void SherpaOnnxCircularBufferFree(const float *p);

// Remove n elements from the buffer
SHERPA_ONNX_API void SherpaOnnxCircularBufferPop(
    const SherpaOnnxCircularBuffer *buffer, int32_t n);

// Return number of elements in the buffer.
SHERPA_ONNX_API int32_t
SherpaOnnxCircularBufferSize(const SherpaOnnxCircularBuffer *buffer);

// Return the head of the buffer. It's always non-decreasing until you
// invoke SherpaOnnxCircularBufferReset() which resets head to 0.
SHERPA_ONNX_API int32_t
SherpaOnnxCircularBufferHead(const SherpaOnnxCircularBuffer *buffer);

// Clear all elements in the buffer
SHERPA_ONNX_API void SherpaOnnxCircularBufferReset(
    const SherpaOnnxCircularBuffer *buffer);

SHERPA_ONNX_API typedef struct SherpaOnnxSpeechSegment {
  // The start index in samples of this segment
  int32_t start;

  // pointer to the array containing the samples
  float *samples;

  // number of samples in this segment
  int32_t n;
} SherpaOnnxSpeechSegment;

typedef struct SherpaOnnxVoiceActivityDetector SherpaOnnxVoiceActivityDetector;

// Return an instance of VoiceActivityDetector.
// The user has to use SherpaOnnxDestroyVoiceActivityDetector() to free
// the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxVoiceActivityDetector *
SherpaOnnxCreateVoiceActivityDetector(const SherpaOnnxVadModelConfig *config,
                                      float buffer_size_in_seconds);

SHERPA_ONNX_API void SherpaOnnxDestroyVoiceActivityDetector(
    const SherpaOnnxVoiceActivityDetector *p);

SHERPA_ONNX_API void SherpaOnnxVoiceActivityDetectorAcceptWaveform(
    const SherpaOnnxVoiceActivityDetector *p, const float *samples, int32_t n);

// Return 1 if there are no speech segments available.
// Return 0 if there are speech segments.
SHERPA_ONNX_API int32_t
SherpaOnnxVoiceActivityDetectorEmpty(const SherpaOnnxVoiceActivityDetector *p);

// Return 1 if there is voice detected.
// Return 0 if voice is silent.
SHERPA_ONNX_API int32_t SherpaOnnxVoiceActivityDetectorDetected(
    const SherpaOnnxVoiceActivityDetector *p);

// Return the first speech segment.
// It throws if SherpaOnnxVoiceActivityDetectorEmpty() returns 1.
SHERPA_ONNX_API void SherpaOnnxVoiceActivityDetectorPop(
    const SherpaOnnxVoiceActivityDetector *p);

// Clear current speech segments.
SHERPA_ONNX_API void SherpaOnnxVoiceActivityDetectorClear(
    const SherpaOnnxVoiceActivityDetector *p);

// Return the first speech segment.
// The user has to use SherpaOnnxDestroySpeechSegment() to free the returned
// pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxSpeechSegment *
SherpaOnnxVoiceActivityDetectorFront(const SherpaOnnxVoiceActivityDetector *p);

// Free the pointer returned SherpaOnnxVoiceActivityDetectorFront().
SHERPA_ONNX_API void SherpaOnnxDestroySpeechSegment(
    const SherpaOnnxSpeechSegment *p);

// Re-initialize the voice activity detector.
SHERPA_ONNX_API void SherpaOnnxVoiceActivityDetectorReset(
    const SherpaOnnxVoiceActivityDetector *p);

SHERPA_ONNX_API void SherpaOnnxVoiceActivityDetectorFlush(
    const SherpaOnnxVoiceActivityDetector *p);

// ============================================================
// For offline Text-to-Speech (i.e., non-streaming TTS)
// ============================================================
SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTtsVitsModelConfig {
  const char *model;
  const char *lexicon;
  const char *tokens;
  const char *data_dir;

  float noise_scale;
  float noise_scale_w;
  float length_scale;    // < 1, faster in speech speed; > 1, slower in speed
  const char *dict_dir;  // unused
} SherpaOnnxOfflineTtsVitsModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTtsMatchaModelConfig {
  const char *acoustic_model;
  const char *vocoder;
  const char *lexicon;
  const char *tokens;
  const char *data_dir;

  float noise_scale;
  float length_scale;    // < 1, faster in speech speed; > 1, slower in speed
  const char *dict_dir;  // unused
} SherpaOnnxOfflineTtsMatchaModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTtsKokoroModelConfig {
  const char *model;
  const char *voices;
  const char *tokens;
  const char *data_dir;

  float length_scale;    // < 1, faster in speech speed; > 1, slower in speed
  const char *dict_dir;  // unused
  const char *lexicon;
  const char *lang;
} SherpaOnnxOfflineTtsKokoroModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTtsKittenModelConfig {
  const char *model;
  const char *voices;
  const char *tokens;
  const char *data_dir;

  float length_scale;  // < 1, faster in speech speed; > 1, slower in speed
} SherpaOnnxOfflineTtsKittenModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTtsZipvoiceModelConfig {
  const char *tokens;
  const char *encoder;
  const char *decoder;
  const char *vocoder;
  const char *data_dir;
  const char *lexicon;
  float feat_scale;
  float t_shift;
  float target_rms;
  float guidance_scale;
} SherpaOnnxOfflineTtsZipvoiceModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTtsModelConfig {
  SherpaOnnxOfflineTtsVitsModelConfig vits;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
  SherpaOnnxOfflineTtsMatchaModelConfig matcha;
  SherpaOnnxOfflineTtsKokoroModelConfig kokoro;
  SherpaOnnxOfflineTtsKittenModelConfig kitten;
  SherpaOnnxOfflineTtsZipvoiceModelConfig zipvoice;
} SherpaOnnxOfflineTtsModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTtsConfig {
  SherpaOnnxOfflineTtsModelConfig model;
  const char *rule_fsts;
  int32_t max_num_sentences;
  const char *rule_fars;
  float silence_scale;
} SherpaOnnxOfflineTtsConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxGeneratedAudio {
  const float *samples;  // in the range [-1, 1]
  int32_t n;             // number of samples
  int32_t sample_rate;
} SherpaOnnxGeneratedAudio;

// If the callback returns 0, then it stops generating
// If the callback returns 1, then it keeps generating
typedef int32_t (*SherpaOnnxGeneratedAudioCallback)(const float *samples,
                                                    int32_t n);

typedef int32_t (*SherpaOnnxGeneratedAudioCallbackWithArg)(const float *samples,
                                                           int32_t n,
                                                           void *arg);

typedef int32_t (*SherpaOnnxGeneratedAudioProgressCallback)(
    const float *samples, int32_t n, float p);

typedef int32_t (*SherpaOnnxGeneratedAudioProgressCallbackWithArg)(
    const float *samples, int32_t n, float p, void *arg);

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTts SherpaOnnxOfflineTts;

// Create an instance of offline TTS. The user has to use DestroyOfflineTts()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOfflineTts *SherpaOnnxCreateOfflineTts(
    const SherpaOnnxOfflineTtsConfig *config);

// Free the pointer returned by SherpaOnnxCreateOfflineTts()
SHERPA_ONNX_API void SherpaOnnxDestroyOfflineTts(
    const SherpaOnnxOfflineTts *tts);

// Return the sample rate of the current TTS object
SHERPA_ONNX_API int32_t
SherpaOnnxOfflineTtsSampleRate(const SherpaOnnxOfflineTts *tts);

// Return the number of speakers of the current TTS object
SHERPA_ONNX_API int32_t
SherpaOnnxOfflineTtsNumSpeakers(const SherpaOnnxOfflineTts *tts);

// Generate audio from the given text and speaker id (sid).
// The user has to use SherpaOnnxDestroyOfflineTtsGeneratedAudio() to free the
// returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerate(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid,
    float speed);

// callback is called whenever SherpaOnnxOfflineTtsConfig.max_num_sentences
// sentences have been processed. The pointer passed to the callback
// is freed once the callback is returned. So the caller should not keep
// a reference to it.
SHERPA_ONNX_API const SherpaOnnxGeneratedAudio *
SherpaOnnxOfflineTtsGenerateWithCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallback callback);

SHERPA_ONNX_API
const SherpaOnnxGeneratedAudio *
SherpaOnnxOfflineTtsGenerateWithProgressCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioProgressCallback callback);

SHERPA_ONNX_API
const SherpaOnnxGeneratedAudio *
SherpaOnnxOfflineTtsGenerateWithProgressCallbackWithArg(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioProgressCallbackWithArg callback, void *arg);

// Same as SherpaOnnxGeneratedAudioCallback but you can pass an additional
// `void* arg` to the callback.
SHERPA_ONNX_API const SherpaOnnxGeneratedAudio *
SherpaOnnxOfflineTtsGenerateWithCallbackWithArg(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallbackWithArg callback, void *arg);

SHERPA_ONNX_API const SherpaOnnxGeneratedAudio *
SherpaOnnxOfflineTtsGenerateWithZipvoice(const SherpaOnnxOfflineTts *tts,
                                         const char *text,
                                         const char *prompt_text,
                                         const float *prompt_samples,
                                         int32_t n_prompt, int32_t prompt_sr,
                                         float speed, int32_t num_steps);

SHERPA_ONNX_API void SherpaOnnxDestroyOfflineTtsGeneratedAudio(
    const SherpaOnnxGeneratedAudio *p);

// Write the generated audio to a wave file.
// The saved wave file contains a single channel and has 16-bit samples.
//
// Return 1 if the write succeeded; return 0 on failure.
SHERPA_ONNX_API int32_t SherpaOnnxWriteWave(const float *samples, int32_t n,
                                            int32_t sample_rate,
                                            const char *filename);

// the amount of bytes needed to store a wave file which contains a
// single channel and has 16-bit samples.
SHERPA_ONNX_API int64_t SherpaOnnxWaveFileSize(int32_t n_samples);

// Similar to SherpaOnnxWriteWave , it writes wave to allocated  buffer;
//
// in some case (http tts api return wave binary file, server do not need to
// write wave to fs)
SHERPA_ONNX_API void SherpaOnnxWriteWaveToBuffer(const float *samples,
                                                 int32_t n, int32_t sample_rate,
                                                 char *buffer);

SHERPA_ONNX_API typedef struct SherpaOnnxWave {
  // samples normalized to the range [-1, 1]
  const float *samples;
  int32_t sample_rate;
  int32_t num_samples;
} SherpaOnnxWave;

// Return a NULL pointer on error. It supports only standard WAVE file.
// Each sample should be 16-bit. It supports only single channel..
//
// If the returned pointer is not NULL, the user has to invoke
// SherpaOnnxFreeWave() to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxWave *SherpaOnnxReadWave(const char *filename);

// Similar to SherpaOnnxReadWave(), it has read the content of `filename`
// into the array `data`.
//
// If the returned pointer is not NULL, the user has to invoke
// SherpaOnnxFreeWave() to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxWave *SherpaOnnxReadWaveFromBinaryData(
    const char *data, int32_t n);

SHERPA_ONNX_API void SherpaOnnxFreeWave(const SherpaOnnxWave *wave);

// ============================================================
// For spoken language identification
// ============================================================

SHERPA_ONNX_API typedef struct
    SherpaOnnxSpokenLanguageIdentificationWhisperConfig {
  const char *encoder;
  const char *decoder;
  int32_t tail_paddings;
} SherpaOnnxSpokenLanguageIdentificationWhisperConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxSpokenLanguageIdentificationConfig {
  SherpaOnnxSpokenLanguageIdentificationWhisperConfig whisper;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
} SherpaOnnxSpokenLanguageIdentificationConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxSpokenLanguageIdentification
    SherpaOnnxSpokenLanguageIdentification;

// Create an instance of SpokenLanguageIdentification.
// The user has to invoke SherpaOnnxDestroySpokenLanguageIdentification()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxSpokenLanguageIdentification *
SherpaOnnxCreateSpokenLanguageIdentification(
    const SherpaOnnxSpokenLanguageIdentificationConfig *config);

SHERPA_ONNX_API void SherpaOnnxDestroySpokenLanguageIdentification(
    const SherpaOnnxSpokenLanguageIdentification *slid);

// The user has to invoke SherpaOnnxDestroyOfflineStream()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API SherpaOnnxOfflineStream *
SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(
    const SherpaOnnxSpokenLanguageIdentification *slid);

SHERPA_ONNX_API typedef struct SherpaOnnxSpokenLanguageIdentificationResult {
  // en for English
  // de for German
  // zh for Chinese
  // es for Spanish
  // ...
  const char *lang;
} SherpaOnnxSpokenLanguageIdentificationResult;

// The user has to invoke SherpaOnnxDestroySpokenLanguageIdentificationResult()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxSpokenLanguageIdentificationResult *
SherpaOnnxSpokenLanguageIdentificationCompute(
    const SherpaOnnxSpokenLanguageIdentification *slid,
    const SherpaOnnxOfflineStream *s);

SHERPA_ONNX_API void SherpaOnnxDestroySpokenLanguageIdentificationResult(
    const SherpaOnnxSpokenLanguageIdentificationResult *r);

// ============================================================
// For speaker embedding extraction
// ============================================================
SHERPA_ONNX_API typedef struct SherpaOnnxSpeakerEmbeddingExtractorConfig {
  const char *model;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
} SherpaOnnxSpeakerEmbeddingExtractorConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxSpeakerEmbeddingExtractor
    SherpaOnnxSpeakerEmbeddingExtractor;

// The user has to invoke SherpaOnnxDestroySpeakerEmbeddingExtractor()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxSpeakerEmbeddingExtractor *
SherpaOnnxCreateSpeakerEmbeddingExtractor(
    const SherpaOnnxSpeakerEmbeddingExtractorConfig *config);

SHERPA_ONNX_API void SherpaOnnxDestroySpeakerEmbeddingExtractor(
    const SherpaOnnxSpeakerEmbeddingExtractor *p);

SHERPA_ONNX_API int32_t SherpaOnnxSpeakerEmbeddingExtractorDim(
    const SherpaOnnxSpeakerEmbeddingExtractor *p);

// The user has to invoke SherpaOnnxDestroyOnlineStream() to free the returned
// pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxOnlineStream *
SherpaOnnxSpeakerEmbeddingExtractorCreateStream(
    const SherpaOnnxSpeakerEmbeddingExtractor *p);

// Return 1 if the stream has enough feature frames for computing embeddings.
// Return 0 otherwise.
SHERPA_ONNX_API int32_t SherpaOnnxSpeakerEmbeddingExtractorIsReady(
    const SherpaOnnxSpeakerEmbeddingExtractor *p,
    const SherpaOnnxOnlineStream *s);

// Compute the embedding of the stream.
//
// @return Return a pointer pointing to an array containing the embedding.
// The length of the array is `dim` as returned by
// SherpaOnnxSpeakerEmbeddingExtractorDim(p)
//
// The user has to invoke SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const float *
SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(
    const SherpaOnnxSpeakerEmbeddingExtractor *p,
    const SherpaOnnxOnlineStream *s);

SHERPA_ONNX_API void SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(
    const float *v);

SHERPA_ONNX_API typedef struct SherpaOnnxSpeakerEmbeddingManager
    SherpaOnnxSpeakerEmbeddingManager;

// The user has to invoke SherpaOnnxDestroySpeakerEmbeddingManager()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxSpeakerEmbeddingManager *
SherpaOnnxCreateSpeakerEmbeddingManager(int32_t dim);

SHERPA_ONNX_API void SherpaOnnxDestroySpeakerEmbeddingManager(
    const SherpaOnnxSpeakerEmbeddingManager *p);

// Register the embedding of a user
//
// @param name  The name of the user
// @param p Pointer to an array containing the embeddings. The length of the
//          array must be equal to `dim` used to construct the manager `p`.
//
// @return Return 1 if added successfully. Return 0 on error
SHERPA_ONNX_API int32_t
SherpaOnnxSpeakerEmbeddingManagerAdd(const SherpaOnnxSpeakerEmbeddingManager *p,
                                     const char *name, const float *v);

// @param v Pointer to an array of embeddings. If there are n embeddings, then
//          v[0] is the pointer to the 0-th array containing the embeddings
//          v[1] is the pointer to the 1-st array containing the embeddings
//          v[n-1] is the pointer to the last array containing the embeddings
//          v[n] is a NULL pointer
// @return Return 1 if added successfully. Return 0 on error
SHERPA_ONNX_API int32_t SherpaOnnxSpeakerEmbeddingManagerAddList(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name,
    const float **v);

// Similar to SherpaOnnxSpeakerEmbeddingManagerAddList() but the memory
// is flattened.
//
// The length of the input array should be `n * dim`.
//
// @return Return 1 if added successfully. Return 0 on error
SHERPA_ONNX_API int32_t SherpaOnnxSpeakerEmbeddingManagerAddListFlattened(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name,
    const float *v, int32_t n);

// Remove a user.
// @param naem The name of the user to remove.
// @return Return 1 if removed successfully; return 0 on error.
//
// Note if the user does not exist, it also returns 0.
SHERPA_ONNX_API int32_t SherpaOnnxSpeakerEmbeddingManagerRemove(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name);

// Search if an existing users' embedding matches the given one.
//
// @param p Pointer to an array containing the embedding. The dim
//          of the array must equal to `dim` used to construct the manager `p`.
// @param threshold A value between 0 and 1. If the similarity score exceeds
//                  this threshold, we say a match is found.
// @return Returns the name of the user if found. Return NULL if not found.
//         If not NULL, the caller has to invoke
//          SherpaOnnxSpeakerEmbeddingManagerFreeSearch() to free the returned
//          pointer to avoid memory leak.
SHERPA_ONNX_API const char *SherpaOnnxSpeakerEmbeddingManagerSearch(
    const SherpaOnnxSpeakerEmbeddingManager *p, const float *v,
    float threshold);

SHERPA_ONNX_API void SherpaOnnxSpeakerEmbeddingManagerFreeSearch(
    const char *name);

SHERPA_ONNX_API typedef struct SherpaOnnxSpeakerEmbeddingManagerSpeakerMatch {
  float score;
  const char *name;
} SherpaOnnxSpeakerEmbeddingManagerSpeakerMatch;

SHERPA_ONNX_API typedef struct
    SherpaOnnxSpeakerEmbeddingManagerBestMatchesResult {
  const SherpaOnnxSpeakerEmbeddingManagerSpeakerMatch *matches;
  int32_t count;
} SherpaOnnxSpeakerEmbeddingManagerBestMatchesResult;

// Get the best matching speakers whose embeddings match the given
// embedding.
//
// @param p Pointer to the SherpaOnnxSpeakerEmbeddingManager instance.
// @param v Pointer to an array containing the embedding vector.
// @param threshold Minimum similarity score required for a match (between 0 and
// 1).
// @param n Number of best matches to retrieve.
// @return Returns a pointer to
// SherpaOnnxSpeakerEmbeddingManagerBestMatchesResult
//         containing the best matches found. Returns NULL if no matches are
//         found. The caller is responsible for freeing the returned pointer
//         using SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches() to
//         avoid memory leaks.
SHERPA_ONNX_API const SherpaOnnxSpeakerEmbeddingManagerBestMatchesResult *
SherpaOnnxSpeakerEmbeddingManagerGetBestMatches(
    const SherpaOnnxSpeakerEmbeddingManager *p, const float *v, float threshold,
    int32_t n);

SHERPA_ONNX_API void SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches(
    const SherpaOnnxSpeakerEmbeddingManagerBestMatchesResult *r);

// Check whether the input embedding matches the embedding of the input
// speaker.
//
// It is for speaker verification.
//
// @param name The target speaker name.
// @param p The input embedding to check.
// @param threshold A value between 0 and 1.
// @return Return 1 if it matches. Otherwise, it returns 0.
SHERPA_ONNX_API int32_t SherpaOnnxSpeakerEmbeddingManagerVerify(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name,
    const float *v, float threshold);

// Return 1 if the user with the name is in the manager.
// Return 0 if the user does not exist.
SHERPA_ONNX_API int32_t SherpaOnnxSpeakerEmbeddingManagerContains(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name);

// Return number of speakers in the manager.
SHERPA_ONNX_API int32_t SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(
    const SherpaOnnxSpeakerEmbeddingManager *p);

// Return the name of all speakers in the manager.
//
// @return Return an array of pointers `ans`. If there are n speakers, then
// - ans[0] contains the name of the 0-th speaker
// - ans[1] contains the name of the 1-st speaker
// - ans[n-1] contains the name of the last speaker
// - ans[n] is NULL
// If there are no users at all, then ans[0] is NULL. In any case,
// `ans` is not NULL.
//
// Each name is NULL-terminated
//
// The caller has to invoke SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const char *const *
SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(
    const SherpaOnnxSpeakerEmbeddingManager *p);

SHERPA_ONNX_API void SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(
    const char *const *names);

// ============================================================
// For audio tagging
// ============================================================
SHERPA_ONNX_API typedef struct
    SherpaOnnxOfflineZipformerAudioTaggingModelConfig {
  const char *model;
} SherpaOnnxOfflineZipformerAudioTaggingModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxAudioTaggingModelConfig {
  SherpaOnnxOfflineZipformerAudioTaggingModelConfig zipformer;
  const char *ced;
  int32_t num_threads;
  int32_t debug;  // true to print debug information of the model
  const char *provider;
} SherpaOnnxAudioTaggingModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxAudioTaggingConfig {
  SherpaOnnxAudioTaggingModelConfig model;
  const char *labels;
  int32_t top_k;
} SherpaOnnxAudioTaggingConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxAudioEvent {
  const char *name;
  int32_t index;
  float prob;
} SherpaOnnxAudioEvent;

SHERPA_ONNX_API typedef struct SherpaOnnxAudioTagging SherpaOnnxAudioTagging;

// The user has to invoke
// SherpaOnnxDestroyAudioTagging()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxAudioTagging *SherpaOnnxCreateAudioTagging(
    const SherpaOnnxAudioTaggingConfig *config);

SHERPA_ONNX_API void SherpaOnnxDestroyAudioTagging(
    const SherpaOnnxAudioTagging *tagger);

// The user has to invoke SherpaOnnxDestroyOfflineStream()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxOfflineStream *
SherpaOnnxAudioTaggingCreateOfflineStream(const SherpaOnnxAudioTagging *tagger);

// Return an array of pointers. The length of the array is top_k + 1.
// If top_k is -1, then config.top_k is used, where config is the config
// used to create the input tagger.
//
// The ans[0]->prob has the largest probability among the array elements
// The last element of the array is a null pointer
//
// The user has to use SherpaOnnxAudioTaggingFreeResults()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxAudioEvent *const *
SherpaOnnxAudioTaggingCompute(const SherpaOnnxAudioTagging *tagger,
                              const SherpaOnnxOfflineStream *s, int32_t top_k);

SHERPA_ONNX_API void SherpaOnnxAudioTaggingFreeResults(
    const SherpaOnnxAudioEvent *const *p);

// ============================================================
// For punctuation
// ============================================================

SHERPA_ONNX_API typedef struct SherpaOnnxOfflinePunctuationModelConfig {
  const char *ct_transformer;
  int32_t num_threads;
  int32_t debug;  // true to print debug information of the model
  const char *provider;
} SherpaOnnxOfflinePunctuationModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflinePunctuationConfig {
  SherpaOnnxOfflinePunctuationModelConfig model;
} SherpaOnnxOfflinePunctuationConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflinePunctuation
    SherpaOnnxOfflinePunctuation;

// The user has to invoke SherpaOnnxDestroyOfflinePunctuation()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxOfflinePunctuation *
SherpaOnnxCreateOfflinePunctuation(
    const SherpaOnnxOfflinePunctuationConfig *config);

SHERPA_ONNX_API void SherpaOnnxDestroyOfflinePunctuation(
    const SherpaOnnxOfflinePunctuation *punct);

// Add punctuations to the input text.
// The user has to invoke SherpaOfflinePunctuationFreeText()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const char *SherpaOfflinePunctuationAddPunct(
    const SherpaOnnxOfflinePunctuation *punct, const char *text);

SHERPA_ONNX_API void SherpaOfflinePunctuationFreeText(const char *text);

SHERPA_ONNX_API typedef struct SherpaOnnxOnlinePunctuationModelConfig {
  const char *cnn_bilstm;
  const char *bpe_vocab;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
} SherpaOnnxOnlinePunctuationModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOnlinePunctuationConfig {
  SherpaOnnxOnlinePunctuationModelConfig model;
} SherpaOnnxOnlinePunctuationConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOnlinePunctuation
    SherpaOnnxOnlinePunctuation;

// Create an online punctuation processor. The user has to invoke
// SherpaOnnxDestroyOnlinePunctuation() to free the returned pointer
// to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxOnlinePunctuation *
SherpaOnnxCreateOnlinePunctuation(
    const SherpaOnnxOnlinePunctuationConfig *config);

// Free a pointer returned by SherpaOnnxCreateOnlinePunctuation()
SHERPA_ONNX_API void SherpaOnnxDestroyOnlinePunctuation(
    const SherpaOnnxOnlinePunctuation *punctuation);

// Add punctuations to the input text. The user has to invoke
// SherpaOnnxOnlinePunctuationFreeText() to free the returned pointer
// to avoid memory leak
SHERPA_ONNX_API const char *SherpaOnnxOnlinePunctuationAddPunct(
    const SherpaOnnxOnlinePunctuation *punctuation, const char *text);

// Free a pointer returned by SherpaOnnxOnlinePunctuationAddPunct()
SHERPA_ONNX_API void SherpaOnnxOnlinePunctuationFreeText(const char *text);

// for resampling
SHERPA_ONNX_API typedef struct SherpaOnnxLinearResampler
    SherpaOnnxLinearResampler;

/*
      float min_freq = min(sampling_rate_in_hz, samp_rate_out_hz);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;
      int32_t lowpass_filter_width = 6;

      You can set filter_cutoff_hz to lowpass_cutoff
      sand set num_zeros to lowpass_filter_width
*/
// The user has to invoke SherpaOnnxDestroyLinearResampler()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxLinearResampler *
SherpaOnnxCreateLinearResampler(int32_t samp_rate_in_hz,
                                int32_t samp_rate_out_hz,
                                float filter_cutoff_hz, int32_t num_zeros);

SHERPA_ONNX_API void SherpaOnnxDestroyLinearResampler(
    const SherpaOnnxLinearResampler *p);

SHERPA_ONNX_API void SherpaOnnxLinearResamplerReset(
    const SherpaOnnxLinearResampler *p);

typedef struct SherpaOnnxResampleOut {
  const float *samples;
  int32_t n;
} SherpaOnnxResampleOut;
// The user has to invoke SherpaOnnxLinearResamplerResampleFree()
// to free the returned pointer to avoid memory leak.
//
// If this is the last segment, you can set flush to 1; otherwise, please
// set flush to 0
SHERPA_ONNX_API const SherpaOnnxResampleOut *SherpaOnnxLinearResamplerResample(
    const SherpaOnnxLinearResampler *p, const float *input, int32_t input_dim,
    int32_t flush);

SHERPA_ONNX_API void SherpaOnnxLinearResamplerResampleFree(
    const SherpaOnnxResampleOut *p);

SHERPA_ONNX_API int32_t SherpaOnnxLinearResamplerResampleGetInputSampleRate(
    const SherpaOnnxLinearResampler *p);

SHERPA_ONNX_API int32_t SherpaOnnxLinearResamplerResampleGetOutputSampleRate(
    const SherpaOnnxLinearResampler *p);

// =========================================================================
// For offline speaker diarization (i.e., non-streaming speaker diarization)
// =========================================================================
SHERPA_ONNX_API typedef struct
    SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig {
  const char *model;
} SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSpeakerSegmentationModelConfig {
  SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig pyannote;
  int32_t num_threads;   // 1
  int32_t debug;         // false
  const char *provider;  // "cpu"
} SherpaOnnxOfflineSpeakerSegmentationModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxFastClusteringConfig {
  // If greater than 0, then threshold is ignored.
  //
  // We strongly recommend that you set it if you know the number of clusters
  // in advance
  int32_t num_clusters;

  // distance threshold.
  //
  // The smaller, the more clusters it will generate.
  // The larger, the fewer clusters it will generate.
  float threshold;
} SherpaOnnxFastClusteringConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSpeakerDiarizationConfig {
  SherpaOnnxOfflineSpeakerSegmentationModelConfig segmentation;
  SherpaOnnxSpeakerEmbeddingExtractorConfig embedding;
  SherpaOnnxFastClusteringConfig clustering;

  // if a segment is less than this value, then it is discarded
  float min_duration_on;  // in seconds

  // if the gap between to segments of the same speaker is less than this value,
  // then these two segments are merged into a single segment.
  // We do this recursively.
  float min_duration_off;  // in seconds
} SherpaOnnxOfflineSpeakerDiarizationConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSpeakerDiarization
    SherpaOnnxOfflineSpeakerDiarization;

// The users has to invoke SherpaOnnxDestroyOfflineSpeakerDiarization()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxOfflineSpeakerDiarization *
SherpaOnnxCreateOfflineSpeakerDiarization(
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config);

// Free the pointer returned by SherpaOnnxCreateOfflineSpeakerDiarization()
SHERPA_ONNX_API void SherpaOnnxDestroyOfflineSpeakerDiarization(
    const SherpaOnnxOfflineSpeakerDiarization *sd);

// Expected sample rate of the input audio samples
SHERPA_ONNX_API int32_t SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(
    const SherpaOnnxOfflineSpeakerDiarization *sd);

// Only config->clustering is used. All other fields are ignored
SHERPA_ONNX_API void SherpaOnnxOfflineSpeakerDiarizationSetConfig(
    const SherpaOnnxOfflineSpeakerDiarization *sd,
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config);

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSpeakerDiarizationResult
    SherpaOnnxOfflineSpeakerDiarizationResult;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSpeakerDiarizationSegment {
  float start;
  float end;
  int32_t speaker;
} SherpaOnnxOfflineSpeakerDiarizationSegment;

SHERPA_ONNX_API int32_t SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r);

SHERPA_ONNX_API int32_t SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r);

// The user has to invoke SherpaOnnxOfflineSpeakerDiarizationDestroySegment()
// to free the returned pointer to avoid memory leak.
//
// The returned pointer is the start address of an array.
// Number of entries in the array equals to the value
// returned by SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments()
SHERPA_ONNX_API const SherpaOnnxOfflineSpeakerDiarizationSegment *
SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r);

SHERPA_ONNX_API void SherpaOnnxOfflineSpeakerDiarizationDestroySegment(
    const SherpaOnnxOfflineSpeakerDiarizationSegment *s);

typedef int32_t (*SherpaOnnxOfflineSpeakerDiarizationProgressCallback)(
    int32_t num_processed_chunks, int32_t num_total_chunks, void *arg);

typedef int32_t (*SherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg)(
    int32_t num_processed_chunks, int32_t num_total_chunks);

// The user has to invoke SherpaOnnxOfflineSpeakerDiarizationDestroyResult()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcess(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n);

// The user has to invoke SherpaOnnxOfflineSpeakerDiarizationDestroyResult()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n, SherpaOnnxOfflineSpeakerDiarizationProgressCallback callback,
    void *arg);

SHERPA_ONNX_API const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n,
    SherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg callback);

SHERPA_ONNX_API void SherpaOnnxOfflineSpeakerDiarizationDestroyResult(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r);

// =========================================================================
// For offline speech enhancement
// =========================================================================
SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig {
  const char *model;
} SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSpeechDenoiserModelConfig {
  SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  int32_t num_threads;
  int32_t debug;  // true to print debug information of the model
  const char *provider;
} SherpaOnnxOfflineSpeechDenoiserModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSpeechDenoiserConfig {
  SherpaOnnxOfflineSpeechDenoiserModelConfig model;
} SherpaOnnxOfflineSpeechDenoiserConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineSpeechDenoiser
    SherpaOnnxOfflineSpeechDenoiser;

// The users has to invoke SherpaOnnxDestroyOfflineSpeechDenoiser()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaOnnxOfflineSpeechDenoiser *
SherpaOnnxCreateOfflineSpeechDenoiser(
    const SherpaOnnxOfflineSpeechDenoiserConfig *config);

// Free the pointer returned by SherpaOnnxCreateOfflineSpeechDenoiser()
SHERPA_ONNX_API void SherpaOnnxDestroyOfflineSpeechDenoiser(
    const SherpaOnnxOfflineSpeechDenoiser *sd);

SHERPA_ONNX_API int32_t SherpaOnnxOfflineSpeechDenoiserGetSampleRate(
    const SherpaOnnxOfflineSpeechDenoiser *sd);

SHERPA_ONNX_API typedef struct SherpaOnnxDenoisedAudio {
  const float *samples;  // in the range [-1, 1]
  int32_t n;             // number of samples
  int32_t sample_rate;
} SherpaOnnxDenoisedAudio;

// Run speech denosing on input samples
// @param samples  A 1-D array containing the input audio samples. Each sample
//           should be in the range [-1, 1].
// @param n  Number of samples
// @param sample_rate Sample rate of the input samples
//
// The user MUST use SherpaOnnxDestroyDenoisedAudio() to free the returned
// pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxDenoisedAudio *
SherpaOnnxOfflineSpeechDenoiserRun(const SherpaOnnxOfflineSpeechDenoiser *sd,
                                   const float *samples, int32_t n,
                                   int32_t sample_rate);

SHERPA_ONNX_API void SherpaOnnxDestroyDenoisedAudio(
    const SherpaOnnxDenoisedAudio *p);

#ifdef __OHOS__

// It is for HarmonyOS
typedef struct NativeResourceManager NativeResourceManager;

SHERPA_ONNX_API const SherpaOnnxOfflineSpeechDenoiser *
SherpaOnnxCreateOfflineSpeechDenoiserOHOS(
    const SherpaOnnxOfflineSpeechDenoiserConfig *config,
    NativeResourceManager *mgr);

/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          SherpaOnnxDestroyOnlineRecognizer() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxOnlineRecognizer *
SherpaOnnxCreateOnlineRecognizerOHOS(
    const SherpaOnnxOnlineRecognizerConfig *config, NativeResourceManager *mgr);

/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          SherpaOnnxDestroyOfflineRecognizer() to free it to avoid memory
//          leak.
SHERPA_ONNX_API const SherpaOnnxOfflineRecognizer *
SherpaOnnxCreateOfflineRecognizerOHOS(
    const SherpaOnnxOfflineRecognizerConfig *config,
    NativeResourceManager *mgr);

// Return an instance of VoiceActivityDetector.
// The user has to use SherpaOnnxDestroyVoiceActivityDetector() to free
// the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaOnnxVoiceActivityDetector *
SherpaOnnxCreateVoiceActivityDetectorOHOS(
    const SherpaOnnxVadModelConfig *config, float buffer_size_in_seconds,
    NativeResourceManager *mgr);

SHERPA_ONNX_API const SherpaOnnxOfflineTts *SherpaOnnxCreateOfflineTtsOHOS(
    const SherpaOnnxOfflineTtsConfig *config, NativeResourceManager *mgr);

SHERPA_ONNX_API const SherpaOnnxSpeakerEmbeddingExtractor *
SherpaOnnxCreateSpeakerEmbeddingExtractorOHOS(
    const SherpaOnnxSpeakerEmbeddingExtractorConfig *config,
    NativeResourceManager *mgr);

SHERPA_ONNX_API const SherpaOnnxKeywordSpotter *
SherpaOnnxCreateKeywordSpotterOHOS(const SherpaOnnxKeywordSpotterConfig *config,
                                   NativeResourceManager *mgr);

SHERPA_ONNX_API const SherpaOnnxOfflineSpeakerDiarization *
SherpaOnnxCreateOfflineSpeakerDiarizationOHOS(
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config,
    NativeResourceManager *mgr);
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // SHERPA_ONNX_C_API_C_API_H_
