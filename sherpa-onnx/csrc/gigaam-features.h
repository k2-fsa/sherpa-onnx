// sherpa-onnx/csrc/gigaam-features.h
//
// Copyright (c)  2026  sherpa-onnx contributors

#ifndef SHERPA_ONNX_CSRC_GIGAAM_FEATURES_H_
#define SHERPA_ONNX_CSRC_GIGAAM_FEATURES_H_

#include <stdint.h>

#include <array>
#include <vector>

namespace sherpa_onnx {

// The shared frontend for the released GigaAM-v4 and Multilingual-v2 RNN-T.
class GigaAMFeatureExtractor {
 public:
  static constexpr int32_t kSampleRate = 16000;
  static constexpr int32_t kNumBins = 64;
  static constexpr int32_t kWindowSize = 320;
  static constexpr int32_t kHopSize = 160;

  GigaAMFeatureExtractor();

  // Returns log mel features in bin-major order (num_bins, num_frames).
  std::vector<float> Compute(const std::vector<float> &samples) const;

 private:
  static constexpr int32_t kNumFrequencies = kWindowSize / 2 + 1;
  std::array<float, kWindowSize> window_;
  std::array<std::array<float, kNumFrequencies>, kNumBins> filterbank_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_GIGAAM_FEATURES_H_
