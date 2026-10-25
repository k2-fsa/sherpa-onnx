// sherpa-onnx/csrc/online-speech-denoiser-voxlock-impl.h
//
// Copyright (c)  2026  Arbaz Siddiqui

#ifndef SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_VOXLOCK_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_VOXLOCK_IMPL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model.h"
#include "sherpa-onnx/csrc/online-speech-denoiser-impl.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/speech-denoiser-voxlock-stream.h"

namespace sherpa_onnx {

/* With a voice sample in the config, it isolates from the first sample.
 *
 * Without one, it passes audio through unchanged until it has heard
 * voice_sample_seconds seconds of speech, takes that speech as the voice
 * sample, and isolates from then on.
 *
 * Speech is 10 ms frames at least 10 dB above the noise floor, the quietest
 * frame of the last 2 seconds. Digital silence (below -90 dBFS) is left out
 * of the floor. At the switch, the stream first runs over the last 250 ms of
 * audio, whose output is dropped, so it does not start from an empty LSTM
 * state, and the isolated audio fades in over 10 ms.
 */
class OnlineSpeechDenoiserVoxLockImpl : public OnlineSpeechDenoiserImpl {
 public:
  explicit OnlineSpeechDenoiserVoxLockImpl(
      const OnlineSpeechDenoiserConfig &config)
      : model_(config.model) {
    Init(config);
  }

  template <typename Manager>
  OnlineSpeechDenoiserVoxLockImpl(Manager *mgr,
                                  const OnlineSpeechDenoiserConfig &config)
      : model_(mgr, config.model) {
    Init(mgr, config);
  }

  DenoisedAudio Run(const float *samples, int32_t n,
                    int32_t sample_rate) override {
    const int32_t model_sample_rate = model_.GetMetaData().sample_rate;

    if (sample_rate <= 0) {
      SHERPA_ONNX_LOGE("Expected sample_rate > 0. Given: %d", sample_rate);
      return {{}, model_sample_rate};
    }

    if (n < 0) {
      SHERPA_ONNX_LOGE("Expected n >= 0. Given: %d", n);
      return {{}, model_sample_rate};
    }

    if (n == 0) {
      return {{}, model_sample_rate};
    }

    if (input_sample_rate_ == -1) {
      input_sample_rate_ = sample_rate;
      if (input_sample_rate_ != model_sample_rate) {
        SHERPA_ONNX_LOGE(
            "Creating a streaming resampler:\n"
            "   in_sample_rate: %d\n"
            "   output_sample_rate: %d\n",
            input_sample_rate_, model_sample_rate);
        resampler_ =
            CreateVoxLockResampler(input_sample_rate_, model_sample_rate);
      }
    } else if (sample_rate != input_sample_rate_) {
      SHERPA_ONNX_LOGE(
          "Streaming denoiser expects a fixed input sample rate. Previous: %d. "
          "Current: %d. Call Flush() or Reset() before changing it.",
          input_sample_rate_, sample_rate);
      return {{}, model_sample_rate};
    }

    std::vector<float> resampled;
    if (resampler_) {
      resampler_->Resample(samples, n, false, &resampled);
    } else {
      resampled.assign(samples, samples + n);
    }

    DenoisedAudio ans;
    ans.sample_rate = model_sample_rate;
    ans.samples = Process(resampled.data(), resampled.size());
    total_output_samples_ += ans.samples.size();
    return ans;
  }

  DenoisedAudio Flush() override {
    DenoisedAudio ans;
    ans.sample_rate = model_.GetMetaData().sample_rate;

    if (resampler_) {
      std::vector<float> tail;
      float dummy = 0;
      resampler_->Resample(&dummy, 0, true, &tail);
      ans.samples = Process(tail.data(), tail.size());
    }

    if (stream_) {
      auto tail = PostProcess(stream_->Flush());
      ans.samples.insert(ans.samples.end(), tail.begin(), tail.end());
    }

    int64_t remaining = total_input_samples_ - total_output_samples_;
    if (remaining < 0) {
      remaining = 0;
    }

    if (ans.samples.size() > static_cast<size_t>(remaining)) {
      ans.samples.resize(static_cast<size_t>(remaining));
    }

    Reset();
    return ans;
  }

  void Reset() override {
    resampler_.reset();
    input_sample_rate_ = -1;
    total_input_samples_ = 0;
    total_output_samples_ = 0;

    voiced_.clear();
    energy_.clear();
    rest_.clear();
    recent_.clear();
    after_.clear();
    num_to_skip_ = 0;

    if (voice_sample_.empty()) {
      stream_.reset();
      num_faded_ = 0;
    } else {
      stream_ = std::make_unique<SpeechDenoiserVoxLockStream>(
          &model_, voice_sample_.data(), voice_sample_.size());
      num_faded_ = fade_length_;
    }
  }

  int32_t GetSampleRate() const override {
    return model_.GetMetaData().sample_rate;
  }

  int32_t GetFrameShiftInSamples() const override {
    return model_.GetMetaData().hop_length;
  }

 private:
  void Init(const OnlineSpeechDenoiserConfig &config) {
    const auto &meta = model_.GetMetaData();
    const auto &filename = config.model.voxlock.voice_sample;
    if (!filename.empty()) {
      voice_sample_ = ReadVoxLockVoiceSample(filename, meta.sample_rate,
                                             meta.voice_sample_seconds);
    }

    InitStream();
  }

  template <typename Manager>
  void Init(Manager *mgr, const OnlineSpeechDenoiserConfig &config) {
    const auto &meta = model_.GetMetaData();
    const auto &filename = config.model.voxlock.voice_sample;
    if (!filename.empty()) {
      voice_sample_ = ReadVoxLockVoiceSample(mgr, filename, meta.sample_rate,
                                             meta.voice_sample_seconds);
    }

    InitStream();
  }

  void InitStream() {
    const auto &meta = model_.GetMetaData();

    frame_length_ = meta.sample_rate / 100;
    warm_up_length_ = meta.sample_rate / 4;
    fade_length_ = frame_length_;
    num_frames_needed_ = meta.voice_sample_seconds * 100;

    Reset();
  }

  // samples are at the model's sample rate
  std::vector<float> Process(const float *samples, int32_t n) {
    total_input_samples_ += n;

    if (!stream_) {
      FindVoiceSample(samples, n);
      return {samples, samples + n};
    }

    if (num_faded_ < fade_length_) {
      after_.insert(after_.end(), samples, samples + n);
    }

    return PostProcess(stream_->Push(samples, n));
  }

  void FindVoiceSample(const float *samples, int32_t n) {
    recent_.insert(recent_.end(), samples, samples + n);
    if (static_cast<int32_t>(recent_.size()) > warm_up_length_) {
      recent_.erase(recent_.begin(), recent_.end() - warm_up_length_);
    }

    rest_.insert(rest_.end(), samples, samples + n);
    const int32_t num_frames =
        static_cast<int32_t>(rest_.size()) / frame_length_;

    for (int32_t i = 0; i != num_frames; ++i) {
      const float *frame = rest_.data() + i * frame_length_;

      double sum = 0;
      for (int32_t k = 0; k != frame_length_; ++k) {
        sum += static_cast<double>(frame[k]) * frame[k];
      }
      float mean = static_cast<float>(sum / frame_length_);
      float db = 10 * std::log10(mean + 1e-10f);

      if (db > kSilenceDb) {
        energy_.push_back(db);
        if (static_cast<int32_t>(energy_.size()) > kNoiseFloorFrames) {
          energy_.pop_front();
        }
      }

      if (static_cast<int32_t>(energy_.size()) >= kMinNoiseFloorFrames &&
          db >= *std::min_element(energy_.begin(), energy_.end()) +
                    kSpeechMarginDb) {
        voiced_.insert(voiced_.end(), frame, frame + frame_length_);
      }
    }

    rest_.erase(rest_.begin(), rest_.begin() + num_frames * frame_length_);

    const int32_t needed = num_frames_needed_ * frame_length_;
    if (static_cast<int32_t>(voiced_.size()) < needed) {
      return;
    }

    stream_ = std::make_unique<SpeechDenoiserVoxLockStream>(
        &model_, voiced_.data(), needed);

    voiced_.clear();
    energy_.clear();

    num_to_skip_ = recent_.size();
    auto warm_up = stream_->Push(recent_.data(), recent_.size());
    num_to_skip_ -= std::min<int32_t>(num_to_skip_, warm_up.size());
  }

  // Drop the output of the warm-up audio and fade in from the input
  std::vector<float> PostProcess(std::vector<float> samples) {
    int32_t cut = std::min<int32_t>(num_to_skip_, samples.size());
    num_to_skip_ -= cut;
    samples.erase(samples.begin(), samples.begin() + cut);

    if (num_faded_ < fade_length_ && !samples.empty()) {
      int32_t n = std::min<int32_t>(samples.size(), fade_length_ - num_faded_);
      n = std::min<int32_t>(n,
                            static_cast<int32_t>(after_.size()) - num_faded_);

      for (int32_t i = 0; i < n; ++i) {
        float r = (static_cast<float>(num_faded_ + i) + 1) / fade_length_;
        samples[i] = r * samples[i] + (1 - r) * after_[num_faded_ + i];
      }

      num_faded_ += n;
      if (num_faded_ >= fade_length_) {
        after_.clear();
      }
    }

    return samples;
  }

 private:
  static constexpr float kSilenceDb = -90;
  static constexpr float kSpeechMarginDb = 10;
  static constexpr int32_t kNoiseFloorFrames = 200;
  static constexpr int32_t kMinNoiseFloorFrames = 20;

  OfflineSpeechDenoiserVoxLockModel model_;
  std::vector<float> voice_sample_;
  std::unique_ptr<SpeechDenoiserVoxLockStream> stream_;

  std::unique_ptr<LinearResample> resampler_;
  int32_t input_sample_rate_ = -1;
  int64_t total_input_samples_ = 0;
  int64_t total_output_samples_ = 0;

  int32_t frame_length_ = 0;
  int32_t warm_up_length_ = 0;
  int32_t fade_length_ = 0;
  int32_t num_frames_needed_ = 0;

  std::vector<float> voiced_;
  std::deque<float> energy_;
  std::vector<float> rest_;
  std::vector<float> recent_;
  std::vector<float> after_;
  int32_t num_to_skip_ = 0;
  int32_t num_faded_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_VOXLOCK_IMPL_H_
