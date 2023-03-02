// sherpa-onnx/csrc/online-recognizer.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_

#include <memory>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/endpoint.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OnlineRecognizerResult {
  std::string text;

  // TODO(fangjun): Add a method to return a json string
  std::string ToString() const { return text; }
};

struct OnlineRecognizerConfig {
  FeatureExtractorConfig feat_config;
  OnlineTransducerModelConfig model_config;
  EndpointConfig endpoint_config;
  bool enable_endpoint = true;

  std::string decoding_method = "greedy_search";
  // now support modified_beam_search and greedy_search

  int32_t max_active_paths = 4;  // used only for modified_beam_search

  OnlineRecognizerConfig() = default;

  OnlineRecognizerConfig(const FeatureExtractorConfig &feat_config,
                         const OnlineTransducerModelConfig &model_config,
                         const EndpointConfig &endpoint_config,
                         bool enable_endpoint,
                         const std::string &decoding_method,
                         int32_t max_active_paths)
      : feat_config(feat_config),
        model_config(model_config),
        endpoint_config(endpoint_config),
        enable_endpoint(enable_endpoint),
        decoding_method(decoding_method),
        max_active_paths(max_active_paths) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class OnlineRecognizer {
 public:
  explicit OnlineRecognizer(const OnlineRecognizerConfig &config);

#if __ANDROID_API__ >= 9
  OnlineRecognizer(AAssetManager *mgr, const OnlineRecognizerConfig &config);
#endif

  ~OnlineRecognizer();

  /// Create a stream for decoding.
  std::unique_ptr<OnlineStream> CreateStream() const;

  /**
   * Return true if the given stream has enough frames for decoding.
   * Return false otherwise
   */
  bool IsReady(OnlineStream *s) const;

  /** Decode a single stream. */
  void DecodeStream(OnlineStream *s) const {
    OnlineStream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode multiple streams in parallel
   *
   * @param ss Pointer array containing streams to be decoded.
   * @param n Number of streams in `ss`.
   */
  void DecodeStreams(OnlineStream **ss, int32_t n) const;

  OnlineRecognizerResult GetResult(OnlineStream *s) const;

  // Return true if we detect an endpoint for this stream.
  // Note: If this function returns true, you usually want to
  // invoke Reset(s).
  bool IsEndpoint(OnlineStream *s) const;

  // Clear the state of this stream. If IsEndpoint(s) returns true,
  // after calling this function, IsEndpoint(s) will return false
  void Reset(OnlineStream *s) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_
