// sherpa-onnx/csrc/offline-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-stream.h"

#include <assert.h>

#include <algorithm>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

void OfflineFeatureExtractorConfig::Register(ParseOptions *po) {
  po->Register("sample-rate", &sampling_rate,
               "Sampling rate of the input waveform. Must match the one "
               "expected by the model. Note: You can have a different "
               "sample rate for the input waveform. We will do resampling "
               "inside the feature extractor");

  po->Register("feat-dim", &feature_dim,
               "Feature dimension. Must match the one expected by the model.");
}

std::string OfflineFeatureExtractorConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineFeatureExtractorConfig(";
  os << "sampling_rate=" << sampling_rate << ", ";
  os << "feature_dim=" << feature_dim << ")";

  return os.str();
}

class OfflineStream::Impl {
 public:
  explicit Impl(const OfflineFeatureExtractorConfig &config) {
    opts_.frame_opts.dither = 0;
    opts_.frame_opts.snip_edges = false;
    opts_.frame_opts.samp_freq = config.sampling_rate;
    opts_.mel_opts.num_bins = config.feature_dim;

    fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
  }

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    if (sampling_rate != opts_.frame_opts.samp_freq) {
      SHERPA_ONNX_LOGE(
          "Creating a resampler:\n"
          "   in_sample_rate: %d\n"
          "   output_sample_rate: %d\n",
          sampling_rate, static_cast<int32_t>(opts_.frame_opts.samp_freq));

      float min_freq =
          std::min<int32_t>(sampling_rate, opts_.frame_opts.samp_freq);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;

      int32_t lowpass_filter_width = 6;
      auto resampler = std::make_unique<LinearResample>(
          sampling_rate, opts_.frame_opts.samp_freq, lowpass_cutoff,
          lowpass_filter_width);
      std::vector<float> samples;
      resampler->Resample(waveform, n, true, &samples);
      fbank_->AcceptWaveform(opts_.frame_opts.samp_freq, samples.data(),
                             samples.size());
      fbank_->InputFinished();
      return;
    }

    fbank_->AcceptWaveform(sampling_rate, waveform, n);
    fbank_->InputFinished();
  }

  int32_t FeatureDim() const { return opts_.mel_opts.num_bins; }

  std::vector<float> GetFrames() const {
    int32_t n = fbank_->NumFramesReady();
    assert(n > 0 && "Please first call AcceptWaveform()");

    int32_t feature_dim = FeatureDim();

    std::vector<float> features(n * feature_dim);

    float *p = features.data();

    for (int32_t i = 0; i != n; ++i) {
      const float *f = fbank_->GetFrame(i);
      std::copy(f, f + feature_dim, p);
      p += feature_dim;
    }

    return features;
  }

  void SetResult(const OfflineRecognitionResult &r) { r_ = r; }

  const OfflineRecognitionResult &GetResult() const { return r_; }

 private:
  std::unique_ptr<knf::OnlineFbank> fbank_;
  knf::FbankOptions opts_;
  OfflineRecognitionResult r_;
};

OfflineStream::OfflineStream(
    const OfflineFeatureExtractorConfig &config /*= {}*/)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineStream::~OfflineStream() = default;

void OfflineStream::AcceptWaveform(int32_t sampling_rate, const float *waveform,
                                   int32_t n) const {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

int32_t OfflineStream::FeatureDim() const { return impl_->FeatureDim(); }

std::vector<float> OfflineStream::GetFrames() const {
  return impl_->GetFrames();
}

void OfflineStream::SetResult(const OfflineRecognitionResult &r) {
  impl_->SetResult(r);
}

const OfflineRecognitionResult &OfflineStream::GetResult() const {
  return impl_->GetResult();
}

}  // namespace sherpa_onnx
