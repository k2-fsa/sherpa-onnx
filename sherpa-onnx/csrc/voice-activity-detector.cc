// sherpa-onnx/csrc/voice-activity-detector.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/voice-activity-detector.h"

#include <queue>
#include <utility>

#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {

class VoiceActivityDetector::Impl {
 public:
  explicit Impl(const VadModelConfig &config, float buffer_size_in_seconds = 60)
      : model_(VadModel::Create(config)),
        config_(config),
        buffer_(buffer_size_in_seconds * config.sample_rate) {}

  void AcceptWaveform(const float *samples, int32_t n) {
    buffer_.Push(samples, n);

    bool is_speech = model_->IsSpeech(samples, n);
    if (is_speech) {
      if (start_ == -1) {
        // beginning of speech
        start_ = buffer_.Tail() - 2 * model_->WindowSize() -
                 model_->MinSpeechDurationSamples();
      }
    } else {
      // non-speech
      if (start_ != -1) {
        // end of speech, save the speech segment
        int32_t end = buffer_.Tail() - model_->MinSilenceDurationSamples();

        std::vector<float> samples = buffer_.Get(start_, end - start_);
        SpeechSegment segment;

        segment.start = start_;
        segment.samples = std::move(samples);

        segments_.push(std::move(segment));

        buffer_.Pop(end - buffer_.Head());
      }

      start_ = -1;
    }
  }

  bool Empty() const { return segments_.empty(); }

  void Pop() { segments_.pop(); }

  const SpeechSegment &Front() const { return segments_.front(); }

  void Reset() {
    std::queue<SpeechSegment>().swap(segments_);

    model_->Reset();
    buffer_.Reset();

    start_ = -1;
  }

  bool IsSpeechDetected() const { return start_ != -1; }

 private:
  std::queue<SpeechSegment> segments_;

  std::unique_ptr<VadModel> model_;
  VadModelConfig config_;
  CircularBuffer buffer_;

  int32_t start_ = -1;
};

VoiceActivityDetector::VoiceActivityDetector(
    const VadModelConfig &config, float buffer_size_in_seconds /*= 60*/)
    : impl_(std::make_unique<Impl>(config, buffer_size_in_seconds)) {}

VoiceActivityDetector::~VoiceActivityDetector() = default;

void VoiceActivityDetector::AcceptWaveform(const float *samples, int32_t n) {
  impl_->AcceptWaveform(samples, n);
}

bool VoiceActivityDetector::Empty() const { return impl_->Empty(); }

void VoiceActivityDetector::Pop() { impl_->Pop(); }

const SpeechSegment &VoiceActivityDetector::Front() const {
  return impl_->Front();
}

void VoiceActivityDetector::Reset() { impl_->Reset(); }

bool VoiceActivityDetector::IsSpeechDetected() const {
  return impl_->IsSpeechDetected();
}

}  // namespace sherpa_onnx
