// sherpa-onnx/csrc/fa-feature-extractor.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/fa-feature-extractor.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-cmvn.h"
#include "kaldi-native-fbank/csrc/feature-matrix.h"
#include "kaldi-native-fbank/csrc/feature-pitch.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

class FaFeatureExtractor::Impl {
 public:
  explicit Impl(const FaFeatureConfig &config) : config_(config) {}

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    if (finished_) {
      SHERPA_ONNX_LOGE(
          "input already finished; ignoring this AcceptWaveform() call");
      return;
    }
    finished_ = true;

    // Convert [-1, 1] input to the kaldi raw-int16 scale: kaldi features
    // (dither, energy floor) assume raw int16 magnitudes.
    std::vector<float> samples(n);
    for (int32_t i = 0; i != n; ++i) {
      samples[i] = waveform[i] * 32768;
    }

    float expected = config_.sample_frequency;
    if (sampling_rate != expected) {
      SHERPA_ONNX_LOGE(
          "Creating a resampler:\n"
          "   in_sample_rate: %d\n"
          "   output_sample_rate: %d\n",
          sampling_rate, static_cast<int32_t>(expected));
      float min_freq = std::min<float>(sampling_rate, expected);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;
      int32_t lowpass_filter_width = 6;
      LinearResample resampler(static_cast<int32_t>(sampling_rate),
                               static_cast<int32_t>(expected), lowpass_cutoff,
                               lowpass_filter_width);
      std::vector<float> resampled;
      resampler.Resample(samples.data(), samples.size(), true, &resampled);
      samples = std::move(resampled);
    }

    Compute(samples);
  }

  int32_t FeatureDim() const { return feature_dim_; }

  int32_t NumFrames() const {
    if (feature_dim_ == 0) {
      return 0;
    }
    return static_cast<int32_t>(frames_.size() / feature_dim_);
  }
  const std::vector<float> &GetFrames() const { return frames_; }

 private:
  // MFCC -> [processed pitch, concatenated] -> utterance CMVN; the
  // splice/LDA-or-deltas step is model-level and applied by ForcedAligner.
  void Compute(const std::vector<float> &samples) {
    knf::MatrixF feats = ComputeMfcc(samples);
    feature_dim_ = feats.cols();
    if (feats.rows() == 0) {
      return;
    }

    if (config_.use_pitch) {
      knf::MatrixF pitch = ComputePitch(samples);
      if (pitch.rows() != feats.rows()) {
        throw std::runtime_error(
            "pitch/mfcc frame count mismatch: " + std::to_string(pitch.rows()) +
            " vs " + std::to_string(feats.rows()));
      }

      knf::MatrixF joined(feats.rows(), feats.cols() + pitch.cols());
      joined.leftCols(feats.cols()) = feats;
      joined.rightCols(pitch.cols()) = pitch;
      feats = std::move(joined);
      feature_dim_ = feats.cols();
    }

    if (config_.uses_cmvn) {
      knf::CmvnOptions cmvn_opts;
      cmvn_opts.norm_vars = false;  // mean-only, like kaldi apply-cmvn
      knf::MatrixD stats = knf::ComputeCmvnStats(feats);
      feats = knf::ApplyCmvn(stats, cmvn_opts, feats);
    }

    frames_.resize(static_cast<size_t>(feats.rows()) * feats.cols());
    std::copy(feats.data(), feats.data() + frames_.size(), frames_.data());
  }

  knf::MatrixF ComputeMfcc(const std::vector<float> &samples) const {
    knf::MfccOptions mfcc_opts;
    mfcc_opts.frame_opts.samp_freq = config_.sample_frequency;
    mfcc_opts.frame_opts.frame_shift_ms = config_.frame_shift;
    mfcc_opts.frame_opts.frame_length_ms = config_.frame_length;
    mfcc_opts.frame_opts.dither = config_.dither;
    mfcc_opts.frame_opts.snip_edges = config_.snip_edges;
    mfcc_opts.frame_opts.preemph_coeff = config_.preemphasis_coefficient;
    mfcc_opts.mel_opts.num_bins = config_.num_mel_bins;
    mfcc_opts.mel_opts.low_freq = config_.low_frequency;
    mfcc_opts.mel_opts.high_freq = config_.high_frequency;
    mfcc_opts.num_ceps = config_.num_coefficients;
    mfcc_opts.cepstral_lifter = config_.cepstral_lifter;
    mfcc_opts.use_energy = config_.use_energy;
    mfcc_opts.energy_floor = config_.energy_floor;
    mfcc_opts.raw_energy = config_.raw_energy;

    knf::OnlineMfcc mfcc(mfcc_opts);
    mfcc.AcceptWaveform(static_cast<int32_t>(config_.sample_frequency),
                        samples.data(), samples.size());
    mfcc.InputFinished();

    int32_t num_frames = mfcc.NumFramesReady();
    int32_t dim = config_.num_coefficients;
    knf::MatrixF feats(num_frames, dim);
    for (int32_t i = 0; i != num_frames; ++i) {
      const float *f = mfcc.GetFrame(i);
      std::copy(f, f + dim, feats.data() + i * dim);
    }
    return feats;
  }

  // Processed pitch: POV + normalized-log-pitch + delta-pitch (3 dims),
  // kaldi's compute-and-process-kaldi-pitch-feats defaults.
  knf::MatrixF ComputePitch(const std::vector<float> &samples) const {
    knf::PitchExtractionOptions pitch_opts;
    pitch_opts.samp_freq = config_.sample_frequency;
    pitch_opts.frame_shift_ms = config_.frame_shift;
    pitch_opts.frame_length_ms = config_.frame_length;
    pitch_opts.snip_edges = config_.snip_edges;
    pitch_opts.min_f0 = config_.min_f0;
    pitch_opts.max_f0 = config_.max_f0;
    pitch_opts.delta_pitch = config_.delta_pitch;
    pitch_opts.penalty_factor = config_.penalty_factor;

    return knf::ComputeAndProcessKaldiPitch(
        pitch_opts, knf::ProcessPitchOptions(), samples.data(),
        static_cast<int32_t>(samples.size()));
  }

 private:
  FaFeatureConfig config_;
  bool finished_ = false;
  int32_t feature_dim_ = 0;
  std::vector<float> frames_;
};

FaFeatureExtractor::FaFeatureExtractor(const FaFeatureConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FaFeatureExtractor::~FaFeatureExtractor() = default;

void FaFeatureExtractor::AcceptWaveform(int32_t sampling_rate,
                                        const float *waveform, int32_t n) {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

int32_t FaFeatureExtractor::FeatureDim() const { return impl_->FeatureDim(); }

int32_t FaFeatureExtractor::NumFrames() const { return impl_->NumFrames(); }

const std::vector<float> &FaFeatureExtractor::GetFrames() const {
  return impl_->GetFrames();
}

}  // namespace sherpa_onnx
