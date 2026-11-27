// sherpa-onnx/csrc/gigaam-features.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/gigaam-features.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "kaldi-native-fbank/csrc/rfft.h"

namespace sherpa_onnx {

GigaAMFeatureExtractor::GigaAMFeatureExtractor() {
  constexpr double kTwoPi = 6.283185307179586;
  for (int32_t i = 0; i < kWindowSize; ++i) {
    window_[i] = 0.5 - 0.5 * std::cos(kTwoPi * i / kWindowSize);
  }

  // HTK mel points, with unnormalized triangular filters in Hz.
  const double max_mel = 2595 * std::log10(1.0 + kSampleRate / 1400.0);
  std::array<double, kNumBins + 2> edges;
  for (int32_t i = 0; i < kNumBins + 2; ++i) {
    const double mel = max_mel * i / (kNumBins + 1);
    edges[i] = 700 * (std::pow(10.0, mel / 2595) - 1);
  }

  for (int32_t bin = 0; bin < kNumBins; ++bin) {
    for (int32_t i = 0; i < kNumFrequencies; ++i) {
      const double hz = i * kSampleRate / static_cast<double>(kWindowSize);
      const double left = (hz - edges[bin]) / (edges[bin + 1] - edges[bin]);
      const double right =
          (edges[bin + 2] - hz) / (edges[bin + 2] - edges[bin + 1]);
      filterbank_[bin][i] = std::max(0.0, std::min(left, right));
    }
  }
}

std::vector<float> GigaAMFeatureExtractor::Compute(
    const std::vector<float> &samples) const {
  if (samples.size() < kWindowSize) {
    return {};
  }

  const size_t frames = 1 + (samples.size() - kWindowSize) / kHopSize;
  std::vector<float> features(kNumBins * frames);
  std::array<float, kWindowSize> spectrum;
  std::array<float, kNumFrequencies> power;
  knf::Rfft fft(kWindowSize);

  for (size_t t = 0; t < frames; ++t) {
    for (int32_t i = 0; i < kWindowSize; ++i) {
      spectrum[i] = samples[t * kHopSize + i] * window_[i];
    }
    fft.Compute(spectrum.data());

    power.front() = spectrum[0] * spectrum[0];
    power.back() = spectrum[1] * spectrum[1];
    for (int32_t i = 1; i < kNumFrequencies - 1; ++i) {
      const float re = spectrum[2 * i];
      const float im = spectrum[2 * i + 1];
      power[i] = re * re + im * im;
    }

    for (int32_t bin = 0; bin < kNumBins; ++bin) {
      float mel = 0;
      for (int32_t i = 0; i < kNumFrequencies; ++i) {
        mel += power[i] * filterbank_[bin][i];
      }
      features[bin * frames + t] = std::log(std::clamp(mel, 1e-9f, 1e9f));
    }
  }
  return features;
}

}  // namespace sherpa_onnx
