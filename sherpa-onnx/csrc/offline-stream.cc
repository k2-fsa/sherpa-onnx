// sherpa-onnx/csrc/offline-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-stream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <utility>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

/* Compute mean and inverse stddev over rows.
 *
 * @param p  A pointer to a 2-d array of shape (num_rows, num_cols)
 * @param num_rows Number of rows
 * @param num_cols Number of columns
 * @param mean On return, it contains p.mean(axis=0)
 * @param inv_stddev On return, it contains 1/p.std(axis=0)
 */
static void ComputeMeanAndInvStd(const float *p, int32_t num_rows,
                                 int32_t num_cols, std::vector<float> *mean,
                                 std::vector<float> *inv_stddev) {
  std::vector<float> sum(num_cols);
  std::vector<float> sum_sq(num_cols);

  for (int32_t i = 0; i != num_rows; ++i) {
    for (int32_t c = 0; c != num_cols; ++c) {
      auto t = p[c];
      sum[c] += t;
      sum_sq[c] += t * t;
    }
    p += num_cols;
  }

  mean->resize(num_cols);
  inv_stddev->resize(num_cols);

  for (int32_t i = 0; i != num_cols; ++i) {
    auto t = sum[i] / num_rows;
    (*mean)[i] = t;

    float stddev = std::sqrt(sum_sq[i] / num_rows - t * t);
    (*inv_stddev)[i] = 1.0f / (stddev + 1e-5f);
  }
}

class OfflineStream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config,
                ContextGraphPtr context_graph)
      : config_(config), context_graph_(std::move(context_graph)) {
    if (config.is_mfcc) {
      mfcc_opts_.frame_opts.dither = config_.dither;
      mfcc_opts_.frame_opts.snip_edges = config_.snip_edges;
      mfcc_opts_.frame_opts.samp_freq = config_.sampling_rate;
      mfcc_opts_.frame_opts.frame_shift_ms = config_.frame_shift_ms;
      mfcc_opts_.frame_opts.frame_length_ms = config_.frame_length_ms;
      mfcc_opts_.frame_opts.remove_dc_offset = config_.remove_dc_offset;
      mfcc_opts_.frame_opts.window_type = config_.window_type;

      mfcc_opts_.mel_opts.num_bins = config_.feature_dim;

      mfcc_opts_.mel_opts.high_freq = config_.high_freq;
      mfcc_opts_.mel_opts.low_freq = config_.low_freq;

      mfcc_opts_.mel_opts.is_librosa = config_.is_librosa;

      mfcc_opts_.num_ceps = config_.num_ceps;
      mfcc_opts_.use_energy = config_.use_energy;

      mfcc_ = std::make_unique<knf::OnlineMfcc>(mfcc_opts_);
    } else {
      opts_.frame_opts.dither = config.dither;
      opts_.frame_opts.snip_edges = config.snip_edges;
      opts_.frame_opts.samp_freq = config.sampling_rate;
      opts_.frame_opts.frame_shift_ms = config.frame_shift_ms;
      opts_.frame_opts.frame_length_ms = config.frame_length_ms;
      opts_.frame_opts.remove_dc_offset = config.remove_dc_offset;
      opts_.frame_opts.window_type = config.window_type;

      opts_.mel_opts.num_bins = config.feature_dim;

      opts_.mel_opts.high_freq = config.high_freq;
      opts_.mel_opts.low_freq = config.low_freq;

      opts_.mel_opts.is_librosa = config.is_librosa;

      fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
    }
  }

  explicit Impl(WhisperTag /*tag*/) {
    config_.normalize_samples = true;
    opts_.frame_opts.samp_freq = 16000;
    opts_.mel_opts.num_bins = 80;  // not used
    whisper_fbank_ =
        std::make_unique<knf::OnlineWhisperFbank>(opts_.frame_opts);
    config_.sampling_rate = opts_.frame_opts.samp_freq;
  }

  explicit Impl(CEDTag /*tag*/) {
    // see
    // https://github.com/RicherMans/CED/blob/main/onnx_inference_with_kaldi.py

    opts_.frame_opts.frame_length_ms = 32;
    opts_.frame_opts.dither = 0;
    opts_.frame_opts.preemph_coeff = 0;
    opts_.frame_opts.remove_dc_offset = false;
    opts_.frame_opts.window_type = "hann";
    opts_.frame_opts.snip_edges = false;

    opts_.frame_opts.samp_freq = 16000;  // fixed to 16000
    opts_.mel_opts.num_bins = 64;
    opts_.mel_opts.high_freq = 8000;

    config_.sampling_rate = opts_.frame_opts.samp_freq;

    fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
  }

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    if (config_.normalize_samples) {
      AcceptWaveformImpl(sampling_rate, waveform, n);
    } else {
      std::vector<float> buf(n);
      for (int32_t i = 0; i != n; ++i) {
        buf[i] = waveform[i] * 32768;
      }
      AcceptWaveformImpl(sampling_rate, buf.data(), n);
    }
  }

  void AcceptWaveformImpl(int32_t sampling_rate, const float *waveform,
                          int32_t n) {
    if (sampling_rate != config_.sampling_rate) {
      SHERPA_ONNX_LOGE(
          "Creating a resampler:\n"
          "   in_sample_rate: %d\n"
          "   output_sample_rate: %d\n",
          sampling_rate, static_cast<int32_t>(config_.sampling_rate));

      float min_freq = std::min<int32_t>(sampling_rate, config_.sampling_rate);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;

      int32_t lowpass_filter_width = 6;
      auto resampler = std::make_unique<LinearResample>(
          sampling_rate, config_.sampling_rate, lowpass_cutoff,
          lowpass_filter_width);
      std::vector<float> samples;
      resampler->Resample(waveform, n, true, &samples);

      if (fbank_) {
        fbank_->AcceptWaveform(config_.sampling_rate, samples.data(),
                               samples.size());
        fbank_->InputFinished();
      } else if (mfcc_) {
        mfcc_->AcceptWaveform(config_.sampling_rate, samples.data(),
                              samples.size());
        mfcc_->InputFinished();
      } else {
        whisper_fbank_->AcceptWaveform(config_.sampling_rate, samples.data(),
                                       samples.size());
        whisper_fbank_->InputFinished();
      }

      return;
    }  // if (sampling_rate != config_.sampling_rate)

    if (fbank_) {
      fbank_->AcceptWaveform(sampling_rate, waveform, n);
      fbank_->InputFinished();
    } else if (mfcc_) {
      mfcc_->AcceptWaveform(sampling_rate, waveform, n);
      mfcc_->InputFinished();
    } else {
      whisper_fbank_->AcceptWaveform(sampling_rate, waveform, n);
      whisper_fbank_->InputFinished();
    }
  }

  int32_t FeatureDim() const {
    return mfcc_ ? mfcc_opts_.num_ceps : opts_.mel_opts.num_bins;
  }

  std::vector<float> GetFrames() const {
    int32_t n = fbank_  ? fbank_->NumFramesReady()
                : mfcc_ ? mfcc_->NumFramesReady()
                        : whisper_fbank_->NumFramesReady();
    assert(n > 0 && "Please first call AcceptWaveform()");

    int32_t feature_dim = FeatureDim();

    std::vector<float> features(n * feature_dim);

    float *p = features.data();

    for (int32_t i = 0; i != n; ++i) {
      const float *f = fbank_  ? fbank_->GetFrame(i)
                       : mfcc_ ? mfcc_->GetFrame(i)
                               : whisper_fbank_->GetFrame(i);
      std::copy(f, f + feature_dim, p);
      p += feature_dim;
    }

    NemoNormalizeFeatures(features.data(), n, feature_dim);

    return features;
  }

  void SetResult(const OfflineRecognitionResult &r) { r_ = r; }

  const OfflineRecognitionResult &GetResult() const { return r_; }

  const ContextGraphPtr &GetContextGraph() const { return context_graph_; }

 private:
  void NemoNormalizeFeatures(float *p, int32_t num_frames,
                             int32_t feature_dim) const {
    if (config_.nemo_normalize_type.empty()) {
      return;
    }

    if (config_.nemo_normalize_type != "per_feature") {
      SHERPA_ONNX_LOGE(
          "Only normalize_type=per_feature is implemented. Given: %s",
          config_.nemo_normalize_type.c_str());
      exit(-1);
    }

    NemoNormalizePerFeature(p, num_frames, feature_dim);
  }

  static void NemoNormalizePerFeature(float *p, int32_t num_frames,
                                      int32_t feature_dim) {
    std::vector<float> mean;
    std::vector<float> inv_stddev;

    ComputeMeanAndInvStd(p, num_frames, feature_dim, &mean, &inv_stddev);

    for (int32_t n = 0; n != num_frames; ++n) {
      for (int32_t i = 0; i != feature_dim; ++i) {
        p[i] = (p[i] - mean[i]) * inv_stddev[i];
      }
      p += feature_dim;
    }
  }

 private:
  FeatureExtractorConfig config_;
  std::unique_ptr<knf::OnlineFbank> fbank_;
  std::unique_ptr<knf::OnlineMfcc> mfcc_;
  std::unique_ptr<knf::OnlineWhisperFbank> whisper_fbank_;
  knf::FbankOptions opts_;
  knf::MfccOptions mfcc_opts_;
  OfflineRecognitionResult r_;
  ContextGraphPtr context_graph_;
};

OfflineStream::OfflineStream(const FeatureExtractorConfig &config /*= {}*/,
                             ContextGraphPtr context_graph /*= nullptr*/)
    : impl_(std::make_unique<Impl>(config, std::move(context_graph))) {}

OfflineStream::OfflineStream(WhisperTag tag)
    : impl_(std::make_unique<Impl>(tag)) {}

OfflineStream::OfflineStream(CEDTag tag) : impl_(std::make_unique<Impl>(tag)) {}

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

const ContextGraphPtr &OfflineStream::GetContextGraph() const {
  return impl_->GetContextGraph();
}

const OfflineRecognitionResult &OfflineStream::GetResult() const {
  return impl_->GetResult();
}
std::string OfflineRecognitionResult::AsJsonString() const {
  std::ostringstream os;
  os << "{";
  os << "\"text\""
     << ": ";
  os << "\"" << text << "\""
     << ", ";

  os << "\""
     << "timestamps"
     << "\""
     << ": ";
  os << "[";

  std::string sep = "";
  for (auto t : timestamps) {
    os << sep << std::fixed << std::setprecision(2) << t;
    sep = ", ";
  }
  os << "], ";

  os << "\""
     << "tokens"
     << "\""
     << ":";
  os << "[";

  sep = "";
  auto oldFlags = os.flags();
  for (const auto &t : tokens) {
    if (t.size() == 1 && static_cast<uint8_t>(t[0]) > 0x7f) {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(t.c_str());
      os << sep << "\""
         << "<0x" << std::hex << std::uppercase << static_cast<uint32_t>(p[0])
         << ">"
         << "\"";
      os.flags(oldFlags);
    } else {
      os << sep << "\"" << t << "\"";
    }
    sep = ", ";
  }
  os << "], ";

  sep = "";

  os << "\""
     << "words"
     << "\""
     << ": ";
  os << "[";
  for (int32_t w : words) {
    os << sep << w;
    sep = ", ";
  }

  os << "]";
  os << "}";

  return os.str();
}
}  // namespace sherpa_onnx
