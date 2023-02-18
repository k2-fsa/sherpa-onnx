// sherpa/csrc/features.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/features.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace sherpa_onnx {

FeatureExtractor::FeatureExtractor() {
  opts_.frame_opts.dither = 0;
  opts_.frame_opts.snip_edges = false;
  opts_.frame_opts.samp_freq = 16000;

  // cache 100 seconds of feature frames, which is more than enough
  // for real needs
  opts_.frame_opts.max_feature_vectors = 100 * 100;

  opts_.mel_opts.num_bins = 80;  // feature dim

  fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
}

FeatureExtractor::FeatureExtractor(const knf::FbankOptions &opts)
    : opts_(opts) {
  fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
}

void FeatureExtractor::AcceptWaveform(float sampling_rate,
                                      const float *waveform, int32_t n) {
  std::lock_guard<std::mutex> lock(mutex_);
  fbank_->AcceptWaveform(sampling_rate, waveform, n);
}

void FeatureExtractor::InputFinished() {
  std::lock_guard<std::mutex> lock(mutex_);
  fbank_->InputFinished();
}

int32_t FeatureExtractor::NumFramesReady() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return fbank_->NumFramesReady();
}

bool FeatureExtractor::IsLastFrame(int32_t frame) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return fbank_->IsLastFrame(frame);
}

std::vector<float> FeatureExtractor::GetFrames(int32_t frame_index,
                                               int32_t n) const {
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

void FeatureExtractor::Reset() {
  fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
}

}  // namespace sherpa_onnx
