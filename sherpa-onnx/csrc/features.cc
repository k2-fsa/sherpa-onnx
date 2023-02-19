// sherpa/csrc/features.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/features.h"

#include <algorithm>
#include <memory>
#include <mutex>  // NOLINT
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace sherpa_onnx {

class FeatureExtractor::Impl {
 public:
  Impl(int32_t sampling_rate, int32_t feature_dim) {
    opts_.frame_opts.dither = 0;
    opts_.frame_opts.snip_edges = false;
    opts_.frame_opts.samp_freq = sampling_rate;

    // cache 100 seconds of feature frames, which is more than enough
    // for real needs
    opts_.frame_opts.max_feature_vectors = 100 * 100;

    opts_.mel_opts.num_bins = feature_dim;

    fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
  }

  void AcceptWaveform(float sampling_rate, const float *waveform, int32_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    fbank_->AcceptWaveform(sampling_rate, waveform, n);
  }

  void InputFinished() {
    std::lock_guard<std::mutex> lock(mutex_);
    fbank_->InputFinished();
  }

  int32_t NumFramesReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fbank_->NumFramesReady();
  }

  bool IsLastFrame(int32_t frame) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fbank_->IsLastFrame(frame);
  }

  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const {
    if (frame_index + n > NumFramesReady()) {
      fprintf(stderr, "%d + %d > %d\n", frame_index, n, NumFramesReady());
      exit(-1);
    }
    std::lock_guard<std::mutex> lock(mutex_);

    int32_t feature_dim = fbank_->Dim();
    std::vector<float> features(feature_dim * n);

    float *p = features.data();

    for (int32_t i = 0; i != n; ++i) {
      const float *f = fbank_->GetFrame(i + frame_index);
      std::copy(f, f + feature_dim, p);
      p += feature_dim;
    }

    return features;
  }

  void Reset() { fbank_ = std::make_unique<knf::OnlineFbank>(opts_); }

  int32_t FeatureDim() const { return opts_.mel_opts.num_bins; }

 private:
  std::unique_ptr<knf::OnlineFbank> fbank_;
  knf::FbankOptions opts_;
  mutable std::mutex mutex_;
};

FeatureExtractor::FeatureExtractor(int32_t sampling_rate /*=16000*/,
                                   int32_t feature_dim /*=80*/)
    : impl_(std::make_unique<Impl>(sampling_rate, feature_dim)) {}

FeatureExtractor::~FeatureExtractor() = default;

void FeatureExtractor::AcceptWaveform(float sampling_rate,
                                      const float *waveform, int32_t n) {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

void FeatureExtractor::InputFinished() { impl_->InputFinished(); }

int32_t FeatureExtractor::NumFramesReady() const {
  return impl_->NumFramesReady();
}

bool FeatureExtractor::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

std::vector<float> FeatureExtractor::GetFrames(int32_t frame_index,
                                               int32_t n) const {
  return impl_->GetFrames(frame_index, n);
}

void FeatureExtractor::Reset() { impl_->Reset(); }

int32_t FeatureExtractor::FeatureDim() const { return impl_->FeatureDim(); }

}  // namespace sherpa_onnx
