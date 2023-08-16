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

#if defined(_WIN32)
#if defined(SHERPA_ONNX_BUILD_SHARED_LIBS)
#define SHERPA_ONNX_EXPORT __declspec(dllexport)
#define SHERPA_ONNX_IMPORT __declspec(dllimport)
#else
#define SHERPA_ONNX_EXPORT
#define SHERPA_ONNX_IMPORT
#endif
#else  // WIN32
#define SHERPA_ONNX_EXPORT
#define SHERPA_ONNX_IMPORT SHERPA_ONNX_EXPORT
#endif

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

SHERPA_ONNX_API typedef struct SherpaOnnxModelConfig {
  SherpaOnnxOnlineTransducerModelConfig transducer;
  SherpaOnnxOnlineParaformerModelConfig paraformer;
  const char *tokens;
  int32_t num_threads;
  const char *provider;
  int32_t debug;  // true to print debug information of the model
  const char *model_type;
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
} SherpaOnnxOnlineRecognizerConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOnlineRecognizerResult {
  // Recognized text
  const char *text;

  // Pointer to continuous memory which holds string based tokens
  // which are seperated by \0
  const char *tokens;

  // a pointer array contains the address of the first item in tokens
  const char *const *tokens_arr;

  // Pointer to continuous memory which holds timestamps
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
//          DestroyOnlineRecognizer() to free it to avoid memory leak.
SHERPA_ONNX_API SherpaOnnxOnlineRecognizer *CreateOnlineRecognizer(
    const SherpaOnnxOnlineRecognizerConfig *config);

/// Free a pointer returned by CreateOnlineRecognizer()
///
/// @param p A pointer returned by CreateOnlineRecognizer()
SHERPA_ONNX_API void DestroyOnlineRecognizer(
    SherpaOnnxOnlineRecognizer *recognizer);

/// Create an online stream for accepting wave samples.
///
/// @param recognizer  A pointer returned by CreateOnlineRecognizer()
/// @return Return a pointer to an OnlineStream. The user has to invoke
///         DestroyOnlineStream() to free it to avoid memory leak.
SHERPA_ONNX_API SherpaOnnxOnlineStream *CreateOnlineStream(
    const SherpaOnnxOnlineRecognizer *recognizer);

/// Destroy an online stream.
///
/// @param stream A pointer returned by CreateOnlineStream()
SHERPA_ONNX_API void DestroyOnlineStream(SherpaOnnxOnlineStream *stream);

/// Accept input audio samples and compute the features.
/// The user has to invoke DecodeOnlineStream() to run the neural network and
/// decoding.
///
/// @param stream  A pointer returned by CreateOnlineStream().
/// @param sample_rate  Sample rate of the input samples. If it is different
///                     from config.feat_config.sample_rate, we will do
///                     resampling inside sherpa-onnx.
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
SHERPA_ONNX_API void AcceptWaveform(SherpaOnnxOnlineStream *stream,
                                    int32_t sample_rate, const float *samples,
                                    int32_t n);

/// Return 1 if there are enough number of feature frames for decoding.
/// Return 0 otherwise.
///
/// @param recognizer  A pointer returned by CreateOnlineRecognizer
/// @param stream  A pointer returned by CreateOnlineStream
SHERPA_ONNX_API int32_t IsOnlineStreamReady(
    SherpaOnnxOnlineRecognizer *recognizer, SherpaOnnxOnlineStream *stream);

/// Call this function to run the neural network model and decoding.
//
/// Precondition for this function: IsOnlineStreamReady() MUST return 1.
///
/// Usage example:
///
///  while (IsOnlineStreamReady(recognizer, stream)) {
///     DecodeOnlineStream(recognizer, stream);
///  }
///
SHERPA_ONNX_API void DecodeOnlineStream(SherpaOnnxOnlineRecognizer *recognizer,
                                        SherpaOnnxOnlineStream *stream);

/// This function is similar to DecodeOnlineStream(). It decodes multiple
/// OnlineStream in parallel.
///
/// Caution: The caller has to ensure each OnlineStream is ready, i.e.,
/// IsOnlineStreamReady() for that stream should return 1.
///
/// @param recognizer  A pointer returned by CreateOnlineRecognizer()
/// @param streams  A pointer array containing pointers returned by
///                 CreateOnlineRecognizer()
/// @param n  Number of elements in the given streams array.
SHERPA_ONNX_API void DecodeMultipleOnlineStreams(
    SherpaOnnxOnlineRecognizer *recognizer, SherpaOnnxOnlineStream **streams,
    int32_t n);

/// Get the decoding results so far for an OnlineStream.
///
/// @param recognizer A pointer returned by CreateOnlineRecognizer().
/// @param stream A pointer returned by CreateOnlineStream().
/// @return A pointer containing the result. The user has to invoke
///         DestroyOnlineRecognizerResult() to free the returned pointer to
///         avoid memory leak.
SHERPA_ONNX_API SherpaOnnxOnlineRecognizerResult *GetOnlineStreamResult(
    SherpaOnnxOnlineRecognizer *recognizer, SherpaOnnxOnlineStream *stream);

/// Destroy the pointer returned by GetOnlineStreamResult().
///
/// @param r A pointer returned by GetOnlineStreamResult()
SHERPA_ONNX_API void DestroyOnlineRecognizerResult(
    const SherpaOnnxOnlineRecognizerResult *r);

/// Reset an OnlineStream , which clears the neural network model state
/// and the state for decoding.
///
/// @param recognizer A pointer returned by CreateOnlineRecognizer().
/// @param stream A pointer returned by CreateOnlineStream
SHERPA_ONNX_API void Reset(SherpaOnnxOnlineRecognizer *recognizer,
                           SherpaOnnxOnlineStream *stream);

/// Signal that no more audio samples would be available.
/// After this call, you cannot call AcceptWaveform() any more.
///
/// @param stream A pointer returned by CreateOnlineStream()
SHERPA_ONNX_API void InputFinished(SherpaOnnxOnlineStream *stream);

/// Return 1 if an endpoint has been detected.
///
/// @param recognizer A pointer returned by CreateOnlineRecognizer()
/// @param stream A pointer returned by CreateOnlineStream()
/// @return Return 1 if an endpoint is detected. Return 0 otherwise.
SHERPA_ONNX_API int32_t IsEndpoint(SherpaOnnxOnlineRecognizer *recognizer,
                                   SherpaOnnxOnlineStream *stream);

// for displaying results on Linux/macOS.
SHERPA_ONNX_API typedef struct SherpaOnnxDisplay SherpaOnnxDisplay;

/// Create a display object. Must be freed using DestroyDisplay to avoid
/// memory leak.
SHERPA_ONNX_API SherpaOnnxDisplay *CreateDisplay(int32_t max_word_per_line);

SHERPA_ONNX_API void DestroyDisplay(SherpaOnnxDisplay *display);

/// Print the result.
SHERPA_ONNX_API void SherpaOnnxPrint(SherpaOnnxDisplay *display, int32_t idx,
                                     const char *s);
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
} SherpaOnnxOfflineWhisperModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineTdnnModelConfig {
  const char *model;
} SherpaOnnxOfflineTdnnModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineLMConfig {
  const char *model;
  float scale;
} SherpaOnnxOfflineLMConfig;

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
} SherpaOnnxOfflineModelConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineRecognizerConfig {
  SherpaOnnxFeatureConfig feat_config;
  SherpaOnnxOfflineModelConfig model_config;
  SherpaOnnxOfflineLMConfig lm_config;

  const char *decoding_method;
  int32_t max_active_paths;
} SherpaOnnxOfflineRecognizerConfig;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineRecognizer
    SherpaOnnxOfflineRecognizer;

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineStream SherpaOnnxOfflineStream;

/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          DestroyOfflineRecognizer() to free it to avoid memory leak.
SHERPA_ONNX_API SherpaOnnxOfflineRecognizer *CreateOfflineRecognizer(
    const SherpaOnnxOfflineRecognizerConfig *config);

/// Free a pointer returned by CreateOfflineRecognizer()
///
/// @param p A pointer returned by CreateOfflineRecognizer()
SHERPA_ONNX_API void DestroyOfflineRecognizer(
    SherpaOnnxOfflineRecognizer *recognizer);

/// Create an offline stream for accepting wave samples.
///
/// @param recognizer  A pointer returned by CreateOfflineRecognizer()
/// @return Return a pointer to an OfflineStream. The user has to invoke
///         DestroyOfflineStream() to free it to avoid memory leak.
SHERPA_ONNX_API SherpaOnnxOfflineStream *CreateOfflineStream(
    const SherpaOnnxOfflineRecognizer *recognizer);

/// Destroy an offline stream.
///
/// @param stream A pointer returned by CreateOfflineStream()
SHERPA_ONNX_API void DestroyOfflineStream(SherpaOnnxOfflineStream *stream);

/// Accept input audio samples and compute the features.
/// The user has to invoke DecodeOfflineStream() to run the neural network and
/// decoding.
///
/// @param stream  A pointer returned by CreateOfflineStream().
/// @param sample_rate  Sample rate of the input samples. If it is different
///                     from config.feat_config.sample_rate, we will do
///                     resampling inside sherpa-onnx.
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
///
/// @caution: For each offline stream, please invoke this function only once!
SHERPA_ONNX_API void AcceptWaveformOffline(SherpaOnnxOfflineStream *stream,
                                           int32_t sample_rate,
                                           const float *samples, int32_t n);
/// Decode an offline stream.
///
/// We assume you have invoked AcceptWaveformOffline() for the given stream
/// before calling this function.
///
/// @param recognizer A pointer returned by CreateOfflineRecognizer().
/// @param stream A pointer returned by CreateOfflineStream()
SHERPA_ONNX_API void DecodeOfflineStream(
    SherpaOnnxOfflineRecognizer *recognizer, SherpaOnnxOfflineStream *stream);

/// Decode a list offline streams in parallel.
///
/// We assume you have invoked AcceptWaveformOffline() for each stream
/// before calling this function.
///
/// @param recognizer A pointer returned by CreateOfflineRecognizer().
/// @param streams A pointer pointer array containing pointers returned
///                by CreateOfflineStream().
/// @param n Number of entries in the given streams.
SHERPA_ONNX_API void DecodeMultipleOfflineStreams(
    SherpaOnnxOfflineRecognizer *recognizer, SherpaOnnxOfflineStream **streams,
    int32_t n);

SHERPA_ONNX_API typedef struct SherpaOnnxOfflineRecognizerResult {
  const char *text;
  // TODO(fangjun): Add more fields
} SherpaOnnxOfflineRecognizerResult;

/// Get the result of the offline stream.
///
/// We assume you have called DecodeOfflineStream() or
/// DecodeMultipleOfflineStreams() with the given stream before calling
/// this function.
///
/// @param stream A pointer returned by CreateOfflineStream().
/// @return Return a pointer to the result. The user has to invoke
///         DestroyOnlineRecognizerResult() to free the returned pointer to
///         avoid memory leak.
SHERPA_ONNX_API SherpaOnnxOfflineRecognizerResult *GetOfflineStreamResult(
    SherpaOnnxOfflineStream *stream);

/// Destroy the pointer returned by GetOfflineStreamResult().
///
/// @param r A pointer returned by GetOfflineStreamResult()
SHERPA_ONNX_API void DestroyOfflineRecognizerResult(
    const SherpaOnnxOfflineRecognizerResult *r);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // SHERPA_ONNX_C_API_C_API_H_
