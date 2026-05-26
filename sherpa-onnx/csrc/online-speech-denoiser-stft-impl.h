// sherpa-onnx/csrc/online-speech-denoiser-stft-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_STFT_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_STFT_IMPL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

struct OnlineSpeechDenoiserStftConfig {
  int32_t sample_rate = 0;
  int32_t n_fft = 0;
  int32_t hop_length = 0;
  int32_t window_length = 0;
  std::string window_type;
};

inline std::vector<float> MakeOnlineSpeechDenoiserWindow(
    const std::string &window_type, int32_t window_length) {
  if (window_type == "vorbis") {
    return MakeVorbisWindow(window_length);
  }

  if (window_type == "hann_sqrt") {
    auto window = knf::GetWindow("hann", window_length);
    for (auto &w : window) {
      w = std::sqrt(w);
    }
    return window;
  }

  return knf::GetWindow(window_type, window_length);
}

class StreamingDft {
 public:
  explicit StreamingDft(int32_t n_fft)
      : n_fft_(n_fft),
        num_bins_(n_fft / 2 + 1),
        cos_f_(num_bins_ * n_fft_),
        sin_f_(num_bins_ * n_fft_),
        cos_i_(n_fft_ * num_bins_),
        sin_i_(n_fft_ * num_bins_) {
    constexpr double kPi = 3.14159265358979323846;
    for (int32_t k = 0; k < num_bins_; ++k) {
      for (int32_t n = 0; n < n_fft_; ++n) {
        double angle = 2.0 * kPi * k * n / n_fft_;
        double c = std::cos(angle);
        double s = std::sin(angle);

        cos_f_[k * n_fft_ + n] = c;
        sin_f_[k * n_fft_ + n] = s;

        cos_i_[n * num_bins_ + k] = c;
        sin_i_[n * num_bins_ + k] = s;
      }
    }
  }

  void Forward(const float *input, float *output) const {
    for (int32_t k = 0; k != num_bins_; ++k) {
      double real = 0;
      double imag = 0;
      const double *p_cos = cos_f_.data() + k * n_fft_;
      const double *p_sin = sin_f_.data() + k * n_fft_;
      for (int32_t n = 0; n != n_fft_; ++n) {
        double v = input[n];
        real += v * p_cos[n];
        imag -= v * p_sin[n];
      }
      output[2 * k] = static_cast<float>(real);
      output[2 * k + 1] = static_cast<float>(imag);
    }
  }

  void Inverse(const float *input, float *output) const {
    for (int32_t n = 0; n != n_fft_; ++n) {
      double sum = input[0];
      if (n_fft_ % 2 == 0) {
        sum += input[2 * (num_bins_ - 1)] * ((n & 1) ? -1.0 : 1.0);
      }

      const double *p_cos = cos_i_.data() + n * num_bins_;
      const double *p_sin = sin_i_.data() + n * num_bins_;
      for (int32_t k = 1; k != num_bins_ - 1; ++k) {
        double real = input[2 * k];
        double imag = input[2 * k + 1];
        sum += 2.0 * (real * p_cos[k] - imag * p_sin[k]);
      }

      output[n] = static_cast<float>(sum / n_fft_);
    }
  }

 private:
  int32_t n_fft_ = 0;
  int32_t num_bins_ = 0;
  std::vector<double> cos_f_;
  std::vector<double> sin_f_;
  std::vector<double> cos_i_;
  std::vector<double> sin_i_;
};

class OnlineSpeechDenoiserStftImpl {
 public:
  explicit OnlineSpeechDenoiserStftImpl(OnlineSpeechDenoiserStftConfig config)
      : config_(std::move(config)),
        fft_(config_.n_fft),
        memory_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        window_(
            MakeOnlineSpeechDenoiserWindow(config_.window_type,
                                           config_.window_length)),
        analysis_buffer_(config_.window_length),
        overlap_add_buffer_(config_.window_length),
        fft_input_(config_.window_length),
        fft_output_(2 * (config_.n_fft / 2 + 1)),
        enhanced_fft_output_(2 * (config_.n_fft / 2 + 1)),
        ifft_output_(config_.window_length),
        zero_hop_(config_.hop_length) {}

  template <typename ProcessFrame>
  DenoisedAudio Run(const float *samples, int32_t n, int32_t sample_rate,
                    ProcessFrame process_frame) {
    if (sample_rate <= 0) {
      SHERPA_ONNX_LOGE("Expected sample_rate > 0. Given: %d", sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    if (n < 0) {
      SHERPA_ONNX_LOGE("Expected n >= 0. Given: %d", n);
      SHERPA_ONNX_EXIT(-1);
    }

    if (n == 0) {
      return {{}, config_.sample_rate};
    }

    if (input_sample_rate_ == -1) {
      input_sample_rate_ = sample_rate;
      CreateResamplerIfNeeded();
    } else if (sample_rate != input_sample_rate_) {
      SHERPA_ONNX_LOGE(
          "Streaming denoiser expects a fixed input sample rate. Previous: %d. "
          "Current: %d.",
          input_sample_rate_, sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<float> resampled;
    if (resampler_) {
      resampler_->Resample(samples, n, false, &resampled);
    } else {
      resampled.assign(samples, samples + n);
    }

    total_input_samples_ += resampled.size();
    pending_input_.insert(pending_input_.end(), resampled.begin(),
                          resampled.end());

    DenoisedAudio ans;
    ans.sample_rate = config_.sample_rate;
    ans.samples = ProcessPending(process_frame);
    total_output_samples_ += ans.samples.size();
    return ans;
  }

  template <typename ProcessFrame, typename ResetModelState>
  DenoisedAudio Flush(ProcessFrame process_frame,
                      ResetModelState reset_model_state) {
    DenoisedAudio ans;
    ans.sample_rate = config_.sample_rate;

    std::vector<float> tail;
    if (resampler_) {
      float dummy = 0;
      resampler_->Resample(&dummy, 0, true, &tail);
      total_input_samples_ += tail.size();
      pending_input_.insert(pending_input_.end(), tail.begin(), tail.end());
    }

    ans.samples = ProcessPending(process_frame);

    if (!pending_input_.empty()) {
      std::vector<float> padded(config_.hop_length, 0.0f);
      std::copy(pending_input_.begin(), pending_input_.end(), padded.begin());
      ProcessHop(padded.data(), &ans.samples, process_frame);
      pending_input_.clear();
    }

    if (started_) {
      ProcessHop(zero_hop_.data(), &ans.samples, process_frame);
    }

    int64_t remaining = total_input_samples_ - total_output_samples_;
    if (remaining < 0) {
      remaining = 0;
    }

    if (ans.samples.size() > static_cast<size_t>(remaining)) {
      ans.samples.resize(static_cast<size_t>(remaining));
    }

    total_output_samples_ += ans.samples.size();
    Reset();
    reset_model_state();
    return ans;
  }

  void Reset() {
    std::fill(analysis_buffer_.begin(), analysis_buffer_.end(), 0.0f);
    std::fill(overlap_add_buffer_.begin(), overlap_add_buffer_.end(), 0.0f);
    pending_input_.clear();
    resampler_.reset();
    input_sample_rate_ = -1;
    started_ = false;
    total_input_samples_ = 0;
    total_output_samples_ = 0;
  }

  int32_t GetSampleRate() const { return config_.sample_rate; }

  int32_t GetFrameShiftInSamples() const { return config_.hop_length; }

  const Ort::MemoryInfo &GetMemoryInfo() const { return memory_info_; }

  int32_t GetNumBins() const { return config_.n_fft / 2 + 1; }

 private:
  void CreateResamplerIfNeeded() {
    if (input_sample_rate_ == config_.sample_rate) {
      return;
    }

    SHERPA_ONNX_LOGE(
        "Creating a streaming resampler:\n"
        "   in_sample_rate: %d\n"
        "   output_sample_rate: %d\n",
        input_sample_rate_, config_.sample_rate);

    float min_freq = std::min<int32_t>(input_sample_rate_, config_.sample_rate);
    float lowpass_cutoff = 0.99f * 0.5f * min_freq;
    int32_t lowpass_filter_width = 6;
    resampler_ = std::make_unique<LinearResample>(
        input_sample_rate_, config_.sample_rate, lowpass_cutoff,
        lowpass_filter_width);
  }

  template <typename ProcessFrame>
  std::vector<float> ProcessPending(ProcessFrame process_frame) {
    std::vector<float> ans;

    int32_t consumed = 0;
    while (static_cast<int32_t>(pending_input_.size()) - consumed >=
           config_.hop_length) {
      ProcessHop(pending_input_.data() + consumed, &ans, process_frame);
      consumed += config_.hop_length;
    }

    if (consumed != 0) {
      pending_input_.erase(pending_input_.begin(),
                           pending_input_.begin() + consumed);
    }

    return ans;
  }

  template <typename ProcessFrame>
  void ProcessHop(const float *hop, std::vector<float> *output,
                  ProcessFrame process_frame) {
    std::move(analysis_buffer_.begin() + config_.hop_length,
              analysis_buffer_.end(), analysis_buffer_.begin());
    std::copy(hop, hop + config_.hop_length,
              analysis_buffer_.end() - config_.hop_length);

    for (int32_t i = 0; i != config_.window_length; ++i) {
      fft_input_[i] = analysis_buffer_[i] * window_[i];
    }

    fft_.Forward(fft_input_.data(), fft_output_.data());
    process_frame(fft_output_.data(), fft_output_.size(),
                  enhanced_fft_output_.data());
    fft_.Inverse(enhanced_fft_output_.data(), ifft_output_.data());

    std::move(overlap_add_buffer_.begin() + config_.hop_length,
              overlap_add_buffer_.end(), overlap_add_buffer_.begin());
    std::fill(overlap_add_buffer_.end() - config_.hop_length,
              overlap_add_buffer_.end(), 0.0f);

    for (int32_t i = 0; i != config_.window_length; ++i) {
      overlap_add_buffer_[i] += ifft_output_[i] * window_[i];
    }

    if (!started_) {
      started_ = true;
      return;
    }

    output->insert(output->end(), overlap_add_buffer_.begin(),
                   overlap_add_buffer_.begin() + config_.hop_length);
  }

 private:
  OnlineSpeechDenoiserStftConfig config_;
  StreamingDft fft_;
  Ort::MemoryInfo memory_info_;

  std::vector<float> window_;
  std::vector<float> analysis_buffer_;
  std::vector<float> overlap_add_buffer_;
  std::vector<float> pending_input_;
  std::vector<float> fft_input_;
  std::vector<float> fft_output_;
  std::vector<float> enhanced_fft_output_;
  std::vector<float> ifft_output_;
  std::vector<float> zero_hop_;
  std::unique_ptr<LinearResample> resampler_;

  int32_t input_sample_rate_ = -1;
  bool started_ = false;
  int64_t total_input_samples_ = 0;
  int64_t total_output_samples_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_STFT_IMPL_H_
