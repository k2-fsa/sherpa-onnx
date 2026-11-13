// sherpa-onnx/csrc/fa-feature-extractor.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FA_FEATURE_EXTRACTOR_H_
#define SHERPA_ONNX_CSRC_FA_FEATURE_EXTRACTOR_H_

#include <stdint.h>

#include <memory>
#include <vector>

namespace sherpa_onnx {

// Feature options of a forced-alignment acoustic model.
// The names and the defaults follow kaldi's feature options
struct FaFeatureConfig {
  float sample_frequency = 16000;
  float frame_shift = 10;   // in ms
  float frame_length = 25;  // in ms
  bool snip_edges = true;
  float dither = 0.0001f;
  bool use_energy = false;
  float energy_floor = 1.0f;
  bool raw_energy = true;
  int32_t num_mel_bins = 23;
  int32_t num_coefficients = 13;
  float cepstral_lifter = 22;
  float low_frequency = 20;
  float high_frequency = 7800;
  float preemphasis_coefficient = 0.97f;
  bool use_pitch = false;  // if true, 3 processed-pitch dims are concatenated
  float min_f0 = 50;
  float max_f0 = 500;
  float delta_pitch = 0.005f;
  float penalty_factor = 0.1f;
  // Utterance-level CMVN: only the mean is normalized, the variance is not.
  bool uses_cmvn = true;
};

// Whole-utterance feature extractor for forced-alignment acoustic models:
//
//   waveform (kaldi int16 scale) -> [resample] -> MFCC -> [processed pitch,
//   concatenated] -> utterance CMVN (mean only, variance untouched)
//
// The trailing step of the feature chain (splice +-N + LDA, or deltas)
// is model-level and is applied by ForcedAligner, not here.
class FaFeatureExtractor {
 public:
  explicit FaFeatureExtractor(const FaFeatureConfig &config);
  ~FaFeatureExtractor();

  // One-shot utterance semantics: the whole feature chain is computed as
  // soon as the samples are accepted. The waveform must be normalized to
  // [-1, 1]; it is converted to the kaldi raw-int16 scale internally and
  // resampled when sampling_rate differs from config.sample_frequency.
  //
  // AcceptWaveform() can be invoked only once; further calls log a warning
  // and are ignored.
  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n);

  // Feature dimension after the chain above (before splice/LDA/deltas).
  int32_t FeatureDim() const;
  int32_t NumFrames() const;

  // All frames as a 1-D array flattened from (num_frames, feature_dim).
  const std::vector<float> &GetFrames() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FA_FEATURE_EXTRACTOR_H_
