// sherpa-onnx/csrc/offline-recognizer.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-lm-config.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/offline-transducer-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineRecognitionResult;

struct OfflineRecognizerConfig {
  OfflineFeatureExtractorConfig feat_config;
  OfflineModelConfig model_config;
  OfflineLMConfig lm_config;

  std::string decoding_method = "greedy_search";
  int32_t max_active_paths = 4;
  // only greedy_search is implemented
  // TODO(fangjun): Implement modified_beam_search

  OfflineRecognizerConfig() = default;
  OfflineRecognizerConfig(const OfflineFeatureExtractorConfig &feat_config,
                          const OfflineModelConfig &model_config,
                          const OfflineLMConfig &lm_config,
                          const std::string &decoding_method,
                          int32_t max_active_paths)
      : feat_config(feat_config),
        model_config(model_config),
        lm_config(lm_config),
        decoding_method(decoding_method),
        max_active_paths(max_active_paths) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class OfflineRecognizerImpl;

class OfflineRecognizer {
 public:
  ~OfflineRecognizer();

  explicit OfflineRecognizer(const OfflineRecognizerConfig &config);

  /// Create a stream for decoding.
  std::unique_ptr<OfflineStream> CreateStream() const;

  /** Decode a single stream
   *
   * @param s The stream to decode.
   */
  void DecodeStream(OfflineStream *s) const {
    OfflineStream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode a list of streams.
   *
   * @param ss Pointer to an array of streams.
   * @param n  Size of the input array.
   */
  void DecodeStreams(OfflineStream **ss, int32_t n) const;

 private:
  std::unique_ptr<OfflineRecognizerImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_H_
