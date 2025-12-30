// sherpa-onnx/c-api/ezy-api.h
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c) 2025 Epicyclism ltd

// C API for sherpa-onnx
//
// Please refer to
// https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/decode-file-c-api.c
// for usages.
//

#ifndef SHERPA_ONNX_EZY_API_H_
#define SHERPA_ONNX_EZY_API_H_

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
  const char *dict_dir;
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

SHERPA_ONNX_API void ResultBasic(
    int32_t *tokens, size_t *count,
                          const SherpaOnnxOnlineRecognizer *recognizer,
                          const SherpaOnnxOnlineStream *stream);

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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // SHERPA_ONNX_EZY_API_H_
