// sherpa-onnx/csrc/speech-denoiser-voxlock-stream.h
//
// Copyright (c)  2026  Arbaz Siddiqui

#ifndef SHERPA_ONNX_CSRC_SPEECH_DENOISER_VOXLOCK_STREAM_H_
#define SHERPA_ONNX_CSRC_SPEECH_DENOISER_VOXLOCK_STREAM_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/wave-reader.h"

namespace sherpa_onnx {

inline std::unique_ptr<LinearResample> CreateVoxLockResampler(
    int32_t in_sample_rate, int32_t out_sample_rate) {
  float min_freq = std::min<int32_t>(in_sample_rate, out_sample_rate);
  float lowpass_cutoff = 0.99f * 0.5f * min_freq;
  int32_t lowpass_filter_width = 6;
  return std::make_unique<LinearResample>(in_sample_rate, out_sample_rate,
                                          lowpass_cutoff, lowpass_filter_width);
}

// Read a voice sample from a wave file and resample it to sample_rate
inline std::vector<float> ReadVoxLockVoiceSample(const std::string &filename,
                                                 int32_t sample_rate) {
  int32_t file_sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples = ReadWave(filename, &file_sample_rate, &is_ok);
  if (!is_ok) {
    SHERPA_ONNX_LOGE("Failed to read the voxlock voice sample '%s'",
                     filename.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  if (file_sample_rate == sample_rate) {
    return samples;
  }

  std::vector<float> ans;
  CreateVoxLockResampler(file_sample_rate, sample_rate)
      ->Resample(samples.data(), samples.size(), true, &ans);
  return ans;
}

/* Return the loudest `seconds` seconds in the first `search_seconds` seconds
 * of the audio, measured over 10 ms frames. If those first `search_seconds`
 * seconds are not longer than `seconds` seconds, it returns all of them.
 *
 * This is the voice sample VoxLock is benchmarked with.
 */
inline std::vector<float> SelectVoxLockVoiceSample(const float *samples,
                                                   int32_t n,
                                                   int32_t sample_rate,
                                                   int32_t seconds,
                                                   int32_t search_seconds) {
  const int32_t num_samples = std::min(n, search_seconds * sample_rate);
  const int32_t frame_length = sample_rate / 100;
  const int32_t num_frames = num_samples / frame_length;
  const int32_t window = seconds * 100;

  if (num_frames <= window) {
    return {samples, samples + num_samples};
  }

  std::vector<float> energy(num_frames);
  for (int32_t i = 0; i != num_frames; ++i) {
    const float *p = samples + i * frame_length;
    double sum = 0;
    for (int32_t k = 0; k != frame_length; ++k) {
      sum += static_cast<double>(p[k]) * p[k];
    }
    energy[i] = static_cast<float>(sum / frame_length);
  }

  int32_t best = 0;
  double best_sum = -1;
  for (int32_t i = 0; i + window <= num_frames; ++i) {
    double sum = 0;
    for (int32_t k = 0; k != window; ++k) {
      sum += energy[i + k];
    }

    if (sum > best_sum) {
      best_sum = sum;
      best = i;
    }
  }

  const float *start = samples + best * frame_length;
  return {start, start + seconds * sample_rate};
}

/* Isolates the caller's voice in a stream of audio at the model's sample
 * rate, locked to the voice in a voice sample.
 *
 * The call is reflect-padded by n_fft / 2 samples at both ends. Every
 * frames_per_chunk frames of the padded call go through the chunk model,
 * which carries the LSTM state across chunks. The frames it returns are
 * overlap-added at hop_length and divided by the summed squared window.
 * The output lags the input by n_fft - hop_length samples until Flush().
 */
class SpeechDenoiserVoxLockStream {
 public:
  SpeechDenoiserVoxLockStream(const OfflineSpeechDenoiserVoxLockModel *model,
                              const float *voice_sample, int32_t n)
      : model_(model),
        n_fft_(model->GetMetaData().n_fft),
        hop_length_(model->GetMetaData().hop_length),
        pad_(model->GetMetaData().n_fft / 2),
        frames_per_chunk_(model->GetMetaData().frames_per_chunk),
        output_gain_(model->GetMetaData().output_gain),
        memory_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        window_square_(n_fft_),
        overlap_add_(n_fft_),
        window_sum_(n_fft_) {
    std::tie(enroll_mag_, context_) = model_->Enroll(voice_sample, n);
    states_ = model_->GetInitStates();

    constexpr double kPi = 3.14159265358979323846;
    for (int32_t i = 0; i != n_fft_; ++i) {
      // periodic Hann window
      float w = static_cast<float>(0.5 - 0.5 * std::cos(2 * kPi * i / n_fft_));
      window_square_[i] = w * w;
    }
  }

  /* @param samples Audio samples at the model's sample rate
   * @param n Number of samples
   *
   * @return Return the isolated samples that are ready.
   */
  std::vector<float> Push(const float *samples, int32_t n) {
    if (!started_) {
      head_.insert(head_.end(), samples, samples + n);
      if (static_cast<int32_t>(head_.size()) <= pad_) {
        return {};
      }

      // reflect padding at the start, as torch.stft(center=True)
      buffer_.reserve(pad_ + head_.size());
      for (int32_t i = pad_; i >= 1; --i) {
        buffer_.push_back(head_[i]);
      }
      buffer_.insert(buffer_.end(), head_.begin(), head_.end());

      head_.clear();
      head_.shrink_to_fit();
      num_to_drop_ = pad_;
      started_ = true;
    } else {
      buffer_.insert(buffer_.end(), samples, samples + n);
    }

    return RunChunks();
  }

  // Return the samples still held back. Call it once, at the end of the
  // stream.
  std::vector<float> Flush() {
    if (!started_) {
      return {};
    }

    // reflect padding at the end
    const int32_t end = static_cast<int32_t>(buffer_.size());
    for (int32_t i = 0; i != pad_; ++i) {
      buffer_.push_back(buffer_[end - 2 - i]);
    }

    std::vector<float> ans = RunChunks();

    for (int32_t i = 0; i != n_fft_; ++i) {
      ans.push_back(Normalize(i) * output_gain_);
    }

    started_ = false;
    buffer_.clear();
    offset_ = 0;

    return ans;
  }

 private:
  float Normalize(int32_t i) const {
    return overlap_add_[i] / (window_sum_[i] > 1e-11f ? window_sum_[i] : 1.0f);
  }

  std::vector<float> RunChunks() {
    std::vector<float> ans;

    while (static_cast<int32_t>(buffer_.size()) - offset_ >= n_fft_) {
      const int32_t available = static_cast<int32_t>(buffer_.size()) - offset_;
      const int32_t num_frames =
          std::min(frames_per_chunk_, 1 + (available - n_fft_) / hop_length_);
      const int32_t num_samples = (num_frames - 1) * hop_length_ + n_fft_;

      std::array<int64_t, 2> wav_shape{1, num_samples};
      Ort::Value wav = Ort::Value::CreateTensor(
          memory_info_, buffer_.data() + offset_, num_samples, wav_shape.data(),
          wav_shape.size());

      Ort::Value frames{nullptr};
      std::tie(frames, states_) =
          model_->Run(std::move(wav), View(&enroll_mag_), View(&context_),
                      std::move(states_));

      if (frames.GetTensorTypeAndShapeInfo().GetElementCount() !=
          static_cast<size_t>(num_frames) * n_fft_) {
        SHERPA_ONNX_LOGE(
            "The voxlock chunk model returned %d values. Expect %d",
            static_cast<int32_t>(
                frames.GetTensorTypeAndShapeInfo().GetElementCount()),
            num_frames * n_fft_);
        SHERPA_ONNX_EXIT(-1);
      }

      const int32_t done = num_frames * hop_length_;
      overlap_add_.resize(n_fft_ + done, 0);
      window_sum_.resize(n_fft_ + done, 0);

      const float *p = frames.GetTensorData<float>();
      for (int32_t j = 0; j != num_frames; ++j) {
        float *dst = overlap_add_.data() + j * hop_length_;
        float *dst_sum = window_sum_.data() + j * hop_length_;
        for (int32_t i = 0; i != n_fft_; ++i) {
          dst[i] += p[j * n_fft_ + i];
          dst_sum[i] += window_square_[i];
        }
      }

      const int32_t cut = std::min(num_to_drop_, done);
      num_to_drop_ -= cut;
      for (int32_t i = cut; i != done; ++i) {
        ans.push_back(Normalize(i) * output_gain_);
      }

      overlap_add_.erase(overlap_add_.begin(), overlap_add_.begin() + done);
      window_sum_.erase(window_sum_.begin(), window_sum_.begin() + done);

      offset_ += done;
    }

    if (offset_ > static_cast<int32_t>(buffer_.size()) / 2) {
      buffer_.erase(buffer_.begin(), buffer_.begin() + offset_);
      offset_ = 0;
    }

    return ans;
  }

 private:
  const OfflineSpeechDenoiserVoxLockModel *model_;
  int32_t n_fft_;
  int32_t hop_length_;
  int32_t pad_;
  int32_t frames_per_chunk_;
  float output_gain_;
  Ort::MemoryInfo memory_info_;

  Ort::Value enroll_mag_{nullptr};
  Ort::Value context_{nullptr};
  OfflineSpeechDenoiserVoxLockModel::States states_;

  std::vector<float> window_square_;

  // Input samples until there are enough to build the padding at the start
  std::vector<float> head_;

  // Padded input; samples before offset_ have been consumed
  std::vector<float> buffer_;
  int32_t offset_ = 0;

  std::vector<float> overlap_add_;
  std::vector<float> window_sum_;

  // Number of output samples that belong to the padding at the start
  int32_t num_to_drop_ = 0;
  bool started_ = false;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEECH_DENOISER_VOXLOCK_STREAM_H_
