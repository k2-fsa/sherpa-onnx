// sherpa-onnx/csrc/vad-stream-state.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/vad-stream-state.h"

#include <algorithm>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

VadStreamState::VadStreamState(const VadModelConfig& config,
                               float buffer_size_in_seconds)
    : buffer_(
          static_cast<int32_t>(buffer_size_in_seconds * config.sample_rate)),
      config_(config),
      sample_rate_(config.sample_rate) {
  if (!config_.silero_vad.model.empty()) {
    max_utterance_length_ =
        config_.sample_rate * config_.silero_vad.max_speech_duration;
  } else if (!config_.ten_vad.model.empty()) {
    max_utterance_length_ =
        config_.sample_rate * config_.ten_vad.max_speech_duration;
  } else {
    SHERPA_ONNX_LOGE("Unsupported VAD model");
    SHERPA_ONNX_EXIT(-1);
  }
}

VadStreamState::~VadStreamState() = default;

void VadStreamState::AcceptWaveform(int32_t sampling_rate,
                                    const float* waveform, int32_t n) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!resampler_ && sampling_rate != sample_rate_) {
    SHERPA_ONNX_LOGE(
        "Creating a resampler:\n"
        "   in_sample_rate: %d\n"
        "   output_sample_rate: %d\n",
        sampling_rate, sample_rate_);

    float min_freq = std::min<int32_t>(sampling_rate, sample_rate_);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler_ = std::make_unique<LinearResample>(
        sampling_rate, sample_rate_, lowpass_cutoff, lowpass_filter_width);
  }

  if (resampler_) {
    if (sampling_rate != resampler_->GetInputSamplingRate()) {
      SHERPA_ONNX_LOGE(
          "You changed the input sampling rate!! Expected: %d, given: %d",
          resampler_->GetInputSamplingRate(), sampling_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<float> samples;
    resampler_->Resample(waveform, n, false, &samples);

    samples_.insert(samples_.end(), samples.begin(), samples.end());
    return;
  }

  samples_.insert(samples_.end(), waveform, waveform + n);
}

bool VadStreamState::IsReady(const VadModel *model) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return IsReadyLocked(model);
}

bool VadStreamState::IsReadyLocked(const VadModel *model) const {
  return static_cast<int64_t>(samples_.size()) - read_pos_ >=
         model->WindowSize();
}

void VadStreamState::Process(VadModel *model) {
  std::lock_guard<std::mutex> lock(mutex_);
  ProcessLocked(model);
}

void VadStreamState::ProcessLocked(VadModel *model) {
  if (!IsReadyLocked(model)) {
    return;
  }

  int32_t window_size = model->WindowSize();
  int32_t window_shift = model->WindowShift();

  // It has the same effect as the check at the beginning of
  // VoiceActivityDetector::AcceptWaveform: once a stream buffers more
  // than the max utterance length, the segment it is building cannot
  // end, so the thresholds relax to split it.
  if (buffer_.Size() > max_utterance_length_) {
    model->SetMinSilenceDuration(0.1);
    model->SetThreshold(0.90);
  } else {
    if (!config_.silero_vad.model.empty()) {
      model->SetMinSilenceDuration(config_.silero_vad.min_silence_duration);
      model->SetThreshold(config_.silero_vad.threshold);
    } else if (!config_.ten_vad.model.empty()) {
      model->SetMinSilenceDuration(config_.ten_vad.min_silence_duration);
      model->SetThreshold(config_.ten_vad.threshold);
    } else {
      SHERPA_ONNX_LOGE("Unknown vad model");
      SHERPA_ONNX_EXIT(-1);
    }
  }

  // The model is shared among streams, so its state travels through this
  // stream: restore it before the window and save it back afterwards.
  // The saved state is empty before the first window, which starts from
  // the model's fresh state, as if the stream had its own instance.
  if (model_state_.states.empty()) {
    model->Reset();
  } else {
    model->SetState(std::move(model_state_));
  }

  const float *p = samples_.data() + read_pos_;
  buffer_.Push(p, window_shift);
  bool is_speech = model->IsSpeech(p, window_size);

  model_state_ = model->GetState();

  read_pos_ += window_shift;

  // Erasing the consumed prefix on every window would move the whole
  // remaining backlog each time, which is quadratic when a caller
  // buffers a long input before processing it. Erase it in bulk instead,
  // once it reaches half of the buffer, so draining stays linear.
  if (read_pos_ >= (1 << 16) &&
      read_pos_ * 2 >= static_cast<int64_t>(samples_.size())) {
    samples_.erase(samples_.begin(),
                   samples_.begin() + static_cast<std::ptrdiff_t>(read_pos_));
    read_pos_ = 0;
  }

  if (is_speech) {
    if (start_ == -1) {
      // beginning of speech
      start_ = std::max(buffer_.Tail() - 2 * model->WindowSize() -
                            model->MinSpeechDurationSamples(),
                        buffer_.Head());
      cur_segment_.start = start_;
    }
    int64_t num_samples = buffer_.Tail() - start_ - 1;
    // num_samples is bounded by the buffer size, so it fits in int32_t.
    cur_segment_.samples =
        buffer_.Get(start_, static_cast<int32_t>(num_samples));
  } else {
    // non-speech

    cur_segment_.start = -1;
    cur_segment_.samples.clear();

    if (start_ != -1 && buffer_.Size()) {
      // end of speech, save the speech segment
      int64_t end = buffer_.Tail() - model->MinSilenceDurationSamples();

      // end - start_ is bounded by the buffer size, so it fits in int32_t.
      std::vector<float> s =
          buffer_.Get(start_, static_cast<int32_t>(end - start_));
      SpeechSegment segment;

      segment.start = start_;
      segment.samples = std::move(s);

      segments_.push(std::move(segment));

      buffer_.Pop(static_cast<int32_t>(end - buffer_.Head()));
    }

    if (start_ == -1) {
      int64_t end = buffer_.Tail() - 2 * model->WindowSize() -
                    model->MinSpeechDurationSamples();
      int32_t n =
          static_cast<int32_t>(std::max<int64_t>(0, end - buffer_.Head()));
      if (n > 0) {
        buffer_.Pop(n);
      }
    }

    start_ = -1;
  }
}

void VadStreamState::Flush(VadModel* model) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (resampler_) {
    std::vector<float> tail;
    resampler_->Resample(nullptr, 0, true, &tail);
    samples_.insert(samples_.end(), tail.begin(), tail.end());
  }

  while (IsReadyLocked(model)) {
    ProcessLocked(model);
  }

  if (start_ == -1 || buffer_.Size() == 0) {
    return;
  }

  // These samples are waiting for a full window and belong to this segment
  buffer_.Push(samples_.data() + read_pos_,
               static_cast<int32_t>(samples_.size() - read_pos_));
  samples_.clear();
  read_pos_ = 0;

  int64_t end = buffer_.Tail();
  if (end <= start_) {
    return;
  }

  // end - start_ is bounded by the buffer size, so it fits in int32_t.
  std::vector<float> s =
      buffer_.Get(start_, static_cast<int32_t>(end - start_));

  SpeechSegment segment;

  segment.start = start_;
  segment.samples = std::move(s);

  segments_.push(std::move(segment));

  buffer_.Pop(static_cast<int32_t>(end - buffer_.Head()));
  start_ = -1;

  cur_segment_.start = -1;
  cur_segment_.samples.clear();
}

void VadStreamState::Reset(VadModel* model) {
  std::lock_guard<std::mutex> lock(mutex_);

  model->Reset();
  model_state_ = model->GetState();

  std::queue<SpeechSegment>().swap(segments_);

  buffer_.Reset();
  samples_.clear();
  read_pos_ = 0;
  resampler_.reset();

  start_ = -1;

  cur_segment_.start = -1;
  cur_segment_.samples.clear();
}

const SpeechSegment &VadStreamState::Front() const {
  static SpeechSegment tmp;

  std::lock_guard<std::mutex> lock(mutex_);

  if (segments_.empty()) {
    SHERPA_ONNX_LOGE(
        "Make sure you call this method only when Empty() returns false; "
        "Return an empty segment");
    return tmp;
  }

  return segments_.front();
}

}  // namespace sherpa_onnx
