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
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
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

/* Return the loudest `seconds` seconds in the first `search_seconds` seconds
 * of the audio, measured over 10 ms frames. It returns an empty vector if
 * the audio is shorter than `seconds` seconds.
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
  const int32_t length = seconds * sample_rate;

  if (num_samples < length) {
    return {};
  }

  if (num_frames <= window) {
    return {samples, samples + length};
  }

  using RowMajorMatrixXf =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  // Mean energy of each 10 ms frame, (num_frames,)
  Eigen::VectorXf energy =
      Eigen::Map<const RowMajorMatrixXf>(samples, num_frames, frame_length)
          .cast<double>()
          .array()
          .square()
          .rowwise()
          .mean()
          .cast<float>();

  // Row i is the window that starts at frame i. The windows overlap, so
  // consecutive rows are one element apart. Only windows whose `length`
  // samples fit in the audio are candidates; frame_length * 100 is less than
  // sample_rate if sample_rate is not a multiple of 100.
  const int32_t num_windows = std::min(
      num_frames - window + 1, (num_samples - length) / frame_length + 1);
  Eigen::Map<const RowMajorMatrixXf, 0, Eigen::Stride<1, 1>> windows(
      energy.data(), num_windows, window);

  // The first window with the largest energy
  Eigen::Index best = 0;
  windows.cast<double>().rowwise().sum().maxCoeff(&best);

  const float *start = samples + best * frame_length;
  return {start, start + length};
}

// A voice sample longer than the model's voice_sample_seconds is searched
// for its loudest voice_sample_seconds seconds within its first
// kVoxLockMaxVoiceSampleSeconds seconds
constexpr int32_t kVoxLockMaxVoiceSampleSeconds = 30;

// Resample a voice sample to sample_rate and return exactly `seconds` seconds
// of it. Return an empty vector if it is shorter than `seconds` seconds.
inline std::vector<float> PrepareVoxLockVoiceSample(
    const std::vector<float> &samples, int32_t file_sample_rate,
    int32_t sample_rate, int32_t seconds, const std::string &filename) {
  std::vector<float> resampled;
  if (file_sample_rate == sample_rate) {
    resampled = samples;
  } else {
    CreateVoxLockResampler(file_sample_rate, sample_rate)
        ->Resample(samples.data(), samples.size(), true, &resampled);
  }

  const int32_t n = resampled.size();
  if (n < seconds * sample_rate) {
    SHERPA_ONNX_LOGE(
        "Ignore the voxlock voice sample '%s': it is %.3f seconds long, "
        "shorter than %d seconds. Take the voice sample from the input audio "
        "instead.",
        filename.c_str(), static_cast<float>(n) / sample_rate, seconds);
    return {};
  }

  if (n > seconds * sample_rate) {
    SHERPA_ONNX_LOGE(
        "The voxlock voice sample '%s' is %.3f seconds long. Use its loudest "
        "%d seconds within its first %d seconds.",
        filename.c_str(), static_cast<float>(n) / sample_rate, seconds,
        kVoxLockMaxVoiceSampleSeconds);
  }

  return SelectVoxLockVoiceSample(resampled.data(), n, sample_rate, seconds,
                                  kVoxLockMaxVoiceSampleSeconds);
}

// Read a voice sample from a wave file and return exactly `seconds` seconds
// of it. Return an empty vector if it cannot be read or is too short.
inline std::vector<float> ReadVoxLockVoiceSample(const std::string &filename,
                                                 int32_t sample_rate,
                                                 int32_t seconds) {
  int32_t file_sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples = ReadWave(filename, &file_sample_rate, &is_ok);
  if (!is_ok) {
    SHERPA_ONNX_LOGE(
        "Ignore the voxlock voice sample '%s': failed to read it. Take the "
        "voice sample from the input audio instead.",
        filename.c_str());
    return {};
  }

  return PrepareVoxLockVoiceSample(samples, file_sample_rate, sample_rate,
                                   seconds, filename);
}

// Read a voice sample from a wave file with a manager, e.g., the asset
// manager on Android, and return exactly `seconds` seconds of it. Return an
// empty vector if it cannot be read or is too short.
template <typename Manager>
std::vector<float> ReadVoxLockVoiceSample(Manager *mgr,
                                          const std::string &filename,
                                          int32_t sample_rate,
                                          int32_t seconds) {
  auto buf = ReadFile(mgr, filename);
  std::istringstream is(std::string(buf.data(), buf.size()));

  int32_t file_sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples = ReadWave(is, &file_sample_rate, &is_ok);
  if (!is_ok) {
    SHERPA_ONNX_LOGE(
        "Ignore the voxlock voice sample '%s': failed to read it. Take the "
        "voice sample from the input audio instead.",
        filename.c_str());
    return {};
  }

  return PrepareVoxLockVoiceSample(samples, file_sample_rate, sample_rate,
                                   seconds, filename);
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
    voice_ = model_->Enroll(voice_sample, n);
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
      // Too short to build the padding at the start; return it unchanged
      return std::exchange(head_, {});
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

      OfflineSpeechDenoiserVoxLockModel::Voice voice;
      voice.reserve(voice_.size());
      for (auto &v : voice_) {
        voice.push_back(View(&v));
      }

      Ort::Value frames{nullptr};
      std::tie(frames, states_) =
          model_->Run(std::move(wav), std::move(voice), std::move(states_));

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
  // Borrowed, not owned by this class. It must outlive this object.
  const OfflineSpeechDenoiserVoxLockModel *model_;
  int32_t n_fft_;
  int32_t hop_length_;
  int32_t pad_;
  int32_t frames_per_chunk_;
  float output_gain_;
  Ort::MemoryInfo memory_info_;

  OfflineSpeechDenoiserVoxLockModel::Voice voice_;
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
