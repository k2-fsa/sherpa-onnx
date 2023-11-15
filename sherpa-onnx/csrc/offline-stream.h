// sherpa-onnx/csrc/offline-stream.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
#define SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineRecognitionResult {
  // Recognition results.
  // For English, it consists of space separated words.
  // For Chinese, it consists of Chinese words without spaces.
  std::string text;

  // Decoded results at the token level.
  // For instance, for BPE-based models it consists of a list of BPE tokens.
  std::vector<std::string> tokens;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  std::vector<float> timestamps;

  std::string AsJsonString() const;
};

struct OfflineFeatureExtractorConfig {
  // Sampling rate used by the feature extractor. If it is different from
  // the sampling rate of the input waveform, we will do resampling inside.
  int32_t sampling_rate = 16000;

  // Feature dimension
  int32_t feature_dim = 80;

  // Set internally by some models, e.g., paraformer and wenet CTC models set
  // it to false.
  // This parameter is not exposed to users from the commandline
  // If true, the feature extractor expects inputs to be normalized to
  // the range [-1, 1].
  // If false, we will multiply the inputs by 32768
  bool normalize_samples = true;

  // For models from NeMo
  // This option is not exposed and is set internally when loading models.
  // Possible values:
  // - per_feature
  // - all_features (not implemented yet)
  // - fixed_mean (not implemented)
  // - fixed_std (not implemented)
  // - or just leave it to empty
  // See
  // https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py#L59
  // for details
  std::string nemo_normalize_type;

  std::string ToString() const;

  void Register(ParseOptions *po);
};

struct WhisperTag {};

class OfflineStream {
 public:
  explicit OfflineStream(const OfflineFeatureExtractorConfig &config = {},
                         ContextGraphPtr context_graph = nullptr);

  explicit OfflineStream(WhisperTag tag,
                         ContextGraphPtr context_graph = nullptr);
  ~OfflineStream();

  /**
     @param sampling_rate The sampling_rate of the input waveform. If it does
                          not equal to  config.sampling_rate, we will do
                          resampling inside.
     @param waveform Pointer to a 1-D array of size n. It must be normalized to
                     the range [-1, 1].
     @param n Number of entries in waveform

     Caution: You can only invoke this function once so you have to input
              all the samples at once
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const;

  /// Return feature dim of this extractor
  int32_t FeatureDim() const;

  // Get all the feature frames of this stream in a 1-D array, which is
  // flattened from a 2-D array of shape (num_frames, feat_dim).
  std::vector<float> GetFrames() const;

  /** Set the recognition result for this stream. */
  void SetResult(const OfflineRecognitionResult &r);

  /** Get the recognition result of this stream */
  const OfflineRecognitionResult &GetResult() const;

  /** Get the ContextGraph of this stream */
  const ContextGraphPtr &GetContextGraph() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
