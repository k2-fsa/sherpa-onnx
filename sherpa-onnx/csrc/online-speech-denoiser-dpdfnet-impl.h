// sherpa-onnx/csrc/online-speech-denoiser-dpdfnet-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_DPDFNET_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_DPDFNET_IMPL_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model.h"
#include "sherpa-onnx/csrc/online-speech-denoiser-impl.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

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

class OnlineSpeechDenoiserDpdfNetImpl : public OnlineSpeechDenoiserImpl {
 public:
  explicit OnlineSpeechDenoiserDpdfNetImpl(
      const OnlineSpeechDenoiserConfig &config)
      : model_(config.model),
        fft_(model_.GetMetaData().n_fft),
        memory_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        window_(MakeVorbisWindow(model_.GetMetaData().window_length)),
        analysis_buffer_(model_.GetMetaData().window_length),
        overlap_add_buffer_(model_.GetMetaData().window_length),
        fft_input_(model_.GetMetaData().window_length),
        fft_output_(model_.GetMetaData().spec_shape[2] *
                    model_.GetMetaData().spec_shape[3]),
        ifft_output_(model_.GetMetaData().window_length),
        zero_hop_(model_.GetMetaData().hop_length),
        state_(model_.GetInitState()) {
    Init();
  }

  template <typename Manager>
  OnlineSpeechDenoiserDpdfNetImpl(Manager *mgr,
                                  const OnlineSpeechDenoiserConfig &config)
      : model_(mgr, config.model),
        fft_(model_.GetMetaData().n_fft),
        memory_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        window_(MakeVorbisWindow(model_.GetMetaData().window_length)),
        analysis_buffer_(model_.GetMetaData().window_length),
        overlap_add_buffer_(model_.GetMetaData().window_length),
        fft_input_(model_.GetMetaData().window_length),
        fft_output_(model_.GetMetaData().spec_shape[2] *
                    model_.GetMetaData().spec_shape[3]),
        ifft_output_(model_.GetMetaData().window_length),
        zero_hop_(model_.GetMetaData().hop_length),
        state_(model_.GetInitState()) {
    Init();
  }

  DenoisedAudio Run(const float *samples, int32_t n, int32_t sample_rate) override {
    const auto &meta = model_.GetMetaData();
    if (sample_rate <= 0) {
      SHERPA_ONNX_LOGE("Expected sample_rate > 0. Given: %d", sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    if (n < 0) {
      SHERPA_ONNX_LOGE("Expected n >= 0. Given: %d", n);
      SHERPA_ONNX_EXIT(-1);
    }

    if (n == 0) {
      return {{}, meta.sample_rate};
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
    ans.sample_rate = meta.sample_rate;
    ans.samples = ProcessPending();
    total_output_samples_ += ans.samples.size();
    return ans;
  }

  DenoisedAudio Flush() override {
    const auto &meta = model_.GetMetaData();
    DenoisedAudio ans;
    ans.sample_rate = meta.sample_rate;

    std::vector<float> tail;
    if (resampler_) {
      float dummy = 0;
      resampler_->Resample(&dummy, 0, true, &tail);
      total_input_samples_ += tail.size();
      pending_input_.insert(pending_input_.end(), tail.begin(), tail.end());
    }

    ans.samples = ProcessPending();

    if (!pending_input_.empty()) {
      std::vector<float> padded(meta.hop_length, 0.0f);
      std::copy(pending_input_.begin(), pending_input_.end(), padded.begin());
      ProcessHop(padded.data(), &ans.samples);
      pending_input_.clear();
    }

    if (started_) {
      ProcessHop(zero_hop_.data(), &ans.samples);
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
    return ans;
  }

  void Reset() override {
    std::fill(analysis_buffer_.begin(), analysis_buffer_.end(), 0.0f);
    std::fill(overlap_add_buffer_.begin(), overlap_add_buffer_.end(), 0.0f);
    pending_input_.clear();
    resampler_.reset();
    input_sample_rate_ = -1;
    started_ = false;
    total_input_samples_ = 0;
    total_output_samples_ = 0;
    state_ = model_.GetInitState();
  }

  int32_t GetSampleRate() const override {
    return model_.GetMetaData().sample_rate;
  }

  int32_t GetFrameShiftInSamples() const override {
    return model_.GetMetaData().hop_length;
  }

 private:
  void Init() {
    const auto &meta = model_.GetMetaData();
    if (meta.profile != "dpdfnet_16khz" &&
        meta.profile != "dpdfnet2_48khz_hr") {
      SHERPA_ONNX_LOGE(
          "Online speech denoiser currently supports only DPDFNet streaming "
          "exports. Given profile: %s",
          meta.profile.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta.spec_shape.size() != 4 || meta.spec_shape[0] != 1 ||
        meta.spec_shape[1] != 1 || meta.spec_shape[3] != 2) {
      SHERPA_ONNX_LOGE(
          "Online speech denoiser expects a single-frame DPDFNet ONNX "
          "signature shaped like [1, 1, F, 2].");
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void CreateResamplerIfNeeded() {
    const auto &meta = model_.GetMetaData();
    if (input_sample_rate_ == meta.sample_rate) {
      return;
    }

    SHERPA_ONNX_LOGE(
        "Creating a streaming resampler:\n"
        "   in_sample_rate: %d\n"
        "   output_sample_rate: %d\n",
        input_sample_rate_, meta.sample_rate);

    float min_freq = std::min<int32_t>(input_sample_rate_, meta.sample_rate);
    float lowpass_cutoff = 0.99f * 0.5f * min_freq;
    int32_t lowpass_filter_width = 6;
    resampler_ = std::make_unique<LinearResample>(
        input_sample_rate_, meta.sample_rate, lowpass_cutoff,
        lowpass_filter_width);
  }

  std::vector<float> ProcessPending() {
    const auto &meta = model_.GetMetaData();
    std::vector<float> ans;

    int32_t consumed = 0;
    while (static_cast<int32_t>(pending_input_.size()) - consumed >=
           meta.hop_length) {
      ProcessHop(pending_input_.data() + consumed, &ans);
      consumed += meta.hop_length;
    }

    if (consumed != 0) {
      pending_input_.erase(pending_input_.begin(),
                           pending_input_.begin() + consumed);
    }

    return ans;
  }

  void ProcessHop(const float *hop, std::vector<float> *output) {
    const auto &meta = model_.GetMetaData();

    std::move(analysis_buffer_.begin() + meta.hop_length, analysis_buffer_.end(),
              analysis_buffer_.begin());
    std::copy(hop, hop + meta.hop_length,
              analysis_buffer_.end() - meta.hop_length);

    for (int32_t i = 0; i != meta.window_length; ++i) {
      fft_input_[i] = analysis_buffer_[i] * window_[i];
    }

    fft_.Forward(fft_input_.data(), fft_output_.data());

    Ort::Value spec = Ort::Value::CreateTensor<float>(
        memory_info_, fft_output_.data(), fft_output_.size(),
        meta.spec_shape.data(), meta.spec_shape.size());

    auto out = model_.Run(std::move(spec), std::move(state_));
    state_ = std::move(out.second);

    const float *enhanced_spec = out.first.GetTensorData<float>();
    fft_.Inverse(enhanced_spec, ifft_output_.data());

    std::move(overlap_add_buffer_.begin() + meta.hop_length,
              overlap_add_buffer_.end(), overlap_add_buffer_.begin());
    std::fill(overlap_add_buffer_.end() - meta.hop_length,
              overlap_add_buffer_.end(), 0.0f);

    for (int32_t i = 0; i != meta.window_length; ++i) {
      overlap_add_buffer_[i] += ifft_output_[i] * window_[i];
    }

    if (!started_) {
      started_ = true;
      return;
    }

    output->insert(output->end(), overlap_add_buffer_.begin(),
                   overlap_add_buffer_.begin() + meta.hop_length);
  }

 private:
  OfflineSpeechDenoiserDpdfNetModel model_;
  StreamingDft fft_;
  Ort::MemoryInfo memory_info_;

  std::vector<float> window_;
  std::vector<float> analysis_buffer_;
  std::vector<float> overlap_add_buffer_;
  std::vector<float> pending_input_;
  std::vector<float> fft_input_;
  std::vector<float> fft_output_;
  std::vector<float> ifft_output_;
  std::vector<float> zero_hop_;
  std::unique_ptr<LinearResample> resampler_;

  int32_t input_sample_rate_ = -1;
  bool started_ = false;
  int64_t total_input_samples_ = 0;
  int64_t total_output_samples_ = 0;

  Ort::Value state_{nullptr};
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_DPDFNET_IMPL_H_
