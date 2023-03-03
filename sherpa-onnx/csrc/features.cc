// sherpa-onnx/csrc/features.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/features.h"

#include <algorithm>
#include <memory>
#include <mutex>  // NOLINT
#include <sstream>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace sherpa_onnx {

void FeatureExtractorConfig::Register(ParseOptions *po) {
  po->Register("sample-rate", &sampling_rate,
               "Sampling rate of the input waveform. Must match the one "
               "expected by the model.");

  po->Register("feat-dim", &feature_dim,
               "Feature dimension. Must match the one expected by the model.");
}

std::string FeatureExtractorConfig::ToString() const {
  std::ostringstream os;

  os << "FeatureExtractorConfig(";
  os << "sampling_rate=" << sampling_rate << ", ";
  os << "feature_dim=" << feature_dim << ", ";
  os << "max_feature_vectors=" << max_feature_vectors << ")";

  return os.str();
}

class FeatureExtractor::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config) {
    opts_.frame_opts.dither = 0;
    opts_.frame_opts.snip_edges = false;
    opts_.frame_opts.samp_freq = config.sampling_rate;

    opts_.frame_opts.max_feature_vectors = config.max_feature_vectors;

    opts_.mel_opts.num_bins = config.feature_dim;

    fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
  }

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
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

  void Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
  }

  int32_t FeatureDim() const { return opts_.mel_opts.num_bins; }

 private:
  std::unique_ptr<knf::OnlineFbank> fbank_;
  knf::FbankOptions opts_;
  mutable std::mutex mutex_;
};

FeatureExtractor::FeatureExtractor(const FeatureExtractorConfig &config /*={}*/)
    : impl_(std::make_unique<Impl>(config)) {}

FeatureExtractor::~FeatureExtractor() = default;

void FeatureExtractor::AcceptWaveform(int32_t sampling_rate,
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
