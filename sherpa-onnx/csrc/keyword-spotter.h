// sherpa-onnx/csrc/keyword-spotter.h
//
// Copyright (c)  2023-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_H_
#define SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/endpoint.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct KeywordResult {
  /// Recognition results.
  /// For English, it consists of space separated words.
  /// For Chinese, it consists of Chinese words without spaces.
  /// Example 1: "hello world"
  /// Example 2: "你好世界"
  std::string keyword;

  /// Decoded results at the token level.
  /// For instance, for BPE-based models it consists of a list of BPE tokens.
  std::vector<std::string> tokens;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  std::vector<float> timestamps;

  /// Starting time of this segment.
  /// When an endpoint is detected, it will change
  float start_time = 0;

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
  std::string AsJsonString() const;
};

struct KeywordSpotterConfig {
  FeatureExtractorConfig feat_config;
  OnlineModelConfig model_config;
  EndpointConfig endpoint_config;
  bool enable_endpoint = true;

  int32_t max_active_paths = 4;

  int32_t num_tailing_blanks = 8;

  float keywords_score = 1.5;

  float keywords_threshold = 0.5;

  std::string keywords_file;

  KeywordSpotterConfig() = default;

  KeywordSpotterConfig(const FeatureExtractorConfig &feat_config,
                       const OnlineModelConfig &model_config,
                       const EndpointConfig &endpoint_config,
                       bool enable_endpoint, int32_t max_active_paths,
                       int32_t num_tailing_blanks, float keywords_score,
                       float keywords_threshold,
                       const std::string &keywords_file)
      : feat_config(feat_config),
        model_config(model_config),
        endpoint_config(endpoint_config),
        enable_endpoint(enable_endpoint),
        max_active_paths(max_active_paths),
        num_tailing_blanks(num_tailing_blanks),
        keywords_score(keywords_score),
        keywords_threshold(keywords_threshold),
        keywords_file(keywords_file) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class KeywordSpotterImpl;

class KeywordSpotter {
 public:
  explicit KeywordSpotter(const KeywordSpotterConfig &config);

#if __ANDROID_API__ >= 9
  KeywordSpotter(AAssetManager *mgr, const KeywordSpotterConfig &config);
#endif

  ~KeywordSpotter();

  /** Create a stream for decoding.
   *
   */
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

  KeywordResult GetResult(OnlineStream *s) const;

  // Return true if we detect an endpoint for this stream.
  // Note: If this function returns true, you usually want to
  // invoke Reset(s).
  bool IsEndpoint(OnlineStream *s) const;

  // Clear the state of this stream. If IsEndpoint(s) returns true,
  // after calling this function, IsEndpoint(s) will return false
  void Reset(OnlineStream *s) const;

 private:
  std::unique_ptr<KeywordSpotterImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_KEYWORD_SPOTTER_H_
