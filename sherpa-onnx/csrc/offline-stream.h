// sherpa-onnx/csrc/offline-stream.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
#define SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineFeatureExtractorConfig {
  int32_t sampling_rate = 16000;
  int32_t feature_dim = 80;

  std::string ToString() const;

  void Register(ParseOptions *po);
};

class OfflineStream {
 public:
  explicit OfflineStream(const OfflineFeatureExtractorConfig &config = {});
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
  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n);

  /// Return feature dim of this extractor
  int32_t FeatureDim() const;

  // Get all the feature frames of this stream in a 1-D array, which is
  // flattened from a 2-D array of shape (num_frames, feat_dim).
  std::vector<float> GetFrames() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_STREAM_H_
