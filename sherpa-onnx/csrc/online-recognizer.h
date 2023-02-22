// sherpa-onnx/csrc/online-recognizer.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/endpoint.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"

namespace sherpa_onnx {

struct OnlineRecognizerResult {
  std::string text;
};

struct OnlineRecognizerConfig {
  FeatureExtractorConfig feat_config;
  OnlineTransducerModelConfig model_config;
  std::string tokens;
  EndpointConfig endpoint_config;
  bool enable_endpoint;

  OnlineRecognizerConfig() = default;

  OnlineRecognizerConfig(const FeatureExtractorConfig &feat_config,
                         const OnlineTransducerModelConfig &model_config,
                         const std::string &tokens,
                         const EndpointConfig &endpoint_config,
                         bool enable_endpoint)
      : feat_config(feat_config),
        model_config(model_config),
        tokens(tokens),
        endpoint_config(endpoint_config),
        enable_endpoint(enable_endpoint) {}

  std::string ToString() const;
};

class OnlineRecognizer {
 public:
  explicit OnlineRecognizer(const OnlineRecognizerConfig &config);
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
