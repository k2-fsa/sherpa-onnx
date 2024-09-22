// sherpa-onnx/csrc/voice-activity-detector.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/voice-activity-detector.h"

#include <algorithm>
#include <queue>
#include <utility>

#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/silero-vad-model.h"

// #define __DEBUG_SPEECH_PROB___
namespace sherpa_onnx {

class timestamp_t {
 public:
  int start;
  int end;

  // default + parameterized constructor
  timestamp_t(int start = -1, int end = -1) : start(start), end(end) {};

  // assignment operator modifies object, therefore non-const
  timestamp_t &operator=(const timestamp_t &a) {
    start = a.start;
    end = a.end;
    return *this;
  };

  // equality comparison. doesn't modify object. therefore const.
  bool operator==(const timestamp_t &a) const {
    return (start == a.start && end == a.end);
  };
};

class VoiceActivityDetector::Impl {
 public:
  explicit Impl(const VadModelConfig &config, float buffer_size_in_seconds = 60)
      : model_(std::make_unique<SileroVadModel>(config)),
        config_(config),
        buffer_((int32_t)(buffer_size_in_seconds * config.sample_rate)) {
    sample_rate = config.sample_rate;
    int32_t sr_per_ms = sample_rate / 1000;
    int32_t speech_pad_ms = 32;

    window_size = model_->WindowSize();
    window_shift = model_->WindowShift();
    threshold = model_->Threshold();

    min_speech_samples = model_->MinSpeechDurationSamples();

    speech_pad_samples = sr_per_ms * speech_pad_ms;

    max_speech_samples = model_->MaxSpeechDurationSamples() - window_shift -
                         2 * speech_pad_samples;

    min_silence_samples = model_->MinSilenceDurationSamples();

    min_silence_samples_at_max_speech = sr_per_ms * 98;

#ifdef __DEBUG_SPEECH_PROB___
    printf(
        "{window_size: %d, min_speech_samples:%d, max_speech_samples:%d, "
        "min_silence_samples:%d,  min_silence_samples_at_max_speech:%d}\n",
        window_size, min_speech_samples, max_speech_samples,
        min_silence_samples, min_silence_samples_at_max_speech);
#endif  //__DEBUG_SPEECH_PROB___
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const VadModelConfig &config,
       float buffer_size_in_seconds = 60)
      : model_(VadModel::Create(mgr, config)),
        config_(config),
        buffer_(buffer_size_in_seconds * config.sample_rate) {}
#endif

  void AcceptWaveform(const float *samples, int32_t n) {
    buffer_.Push(samples, n);
    last_.insert(last_.end(), samples, samples + n);
    if (last_.size() < window_size) {
      return;
    }

    // Note: For v4, window_shift == window_size
    int32_t k =
        (static_cast<int32_t>(last_.size()) - window_size) / window_shift + 1;
    const float *p = last_.data();

    for (int32_t i = 0; i < k; ++i, p += window_shift) {
      float speech_prob = model_->Run(p, window_size);
      current_sample += window_shift;
      // 语音分片
      // Voice fragmentation
      if ((speech_prob >= threshold)) {
#ifdef __DEBUG_SPEECH_PROB___
        float speech =
            current_sample - window_shift;  // minus window_shift to get precise
                                            // start time point.
        printf("{ start: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate,
               speech_prob, current_sample - window_shift);
#endif  //__DEBUG_SPEECH_PROB___
        // 临时结束点重置
        // Temporary end point reset
        if (temp_end != 0) {
          temp_end = 0;
          // 下次的预计开始点小于上次的结束点，重置
          // The next estimated start point is less than the last end point,
          // reset
          if (next_start < prev_end) next_start = current_sample - window_shift;
        }
        // 第一次语音分片,记录开始点
        // First voice segmentation, record start point
        if (triggered == false) {
          triggered = true;
          current_speech.start = current_sample - window_shift;
        }
        continue;
      }

      if (
          // 大于语音分片的最大采样数，强制分片
          // If the number of samples is greater than the maximum number of
          // voice fragments, forced fragmentation
          (triggered == true) &&
          ((current_sample - current_speech.start) > max_speech_samples)) {
        if (prev_end > 0) {
          current_speech.end = prev_end;
#ifdef __DEBUG_SPEECH_PROB___
          printf("{>max_prev speech start: %d, end:%d}\n", current_speech.start,
                 current_speech.end);
#endif  //__DEBUG_SPEECH_PROB___
          std::vector<float> s = buffer_.Get(
              current_speech.start, current_speech.end - current_speech.start);
          SpeechSegment segment;
          segment.start = current_speech.start;
          segment.samples = std::move(s);
          segments_.push(std::move(segment));
          current_speech = timestamp_t();
          // previously reached silence(< neg_thres) and is still not speech(<
          // thres)
          if (next_start < prev_end)
            triggered = false;
          else {
            current_speech.start = next_start;
          }
          prev_end = 0;
          next_start = 0;
          temp_end = 0;
        } else {
          current_speech.end = current_sample;
#ifdef __DEBUG_SPEECH_PROB___
          printf("{>max speech start: %d, end:%d}\n", current_speech.start,
                 current_speech.end);
#endif  //__DEBUG_SPEECH_PROB___
          std::vector<float> s = buffer_.Get(
              current_speech.start, current_speech.end - current_speech.start);
          SpeechSegment segment;
          segment.start = current_speech.start;
          segment.samples = std::move(s);
          segments_.push(std::move(segment));
          current_speech = timestamp_t();
          prev_end = 0;
          next_start = 0;
          temp_end = 0;
          triggered = false;
        }
        continue;
      }
      // 混沌状态，保持原状
      if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold)) {
        if (triggered) {
#ifdef __DEBUG_SPEECH_PROB___
          float speech =
              current_sample - window_shift;  // minus window_shift to get
                                              // precise start time point.
          printf("{ speaking: %.3f s (%.3f) %08d}\n",
                 1.0 * speech / sample_rate, speech_prob,
                 current_sample - window_shift);
#endif  //__DEBUG_SPEECH_PROB___
        } else {
#ifdef __DEBUG_SPEECH_PROB___
          float speech =
              current_sample - window_shift;  // minus window_shift to get
                                              // precise start time point.
          printf("{ silence: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate,
                 speech_prob, current_sample - window_shift);
#endif  //__DEBUG_SPEECH_PROB___
        }
        continue;
      }

      // 4) End
      if ((speech_prob < (threshold - 0.15))) {
#ifdef __DEBUG_SPEECH_PROB___
        float speech = current_sample - window_shift -
                       speech_pad_samples;  // minus window_shift to get precise
                                            // start time point.
        if (speech < 0.0f) {
          speech = 0.0f;
        }
        printf("{ end: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate,
               speech_prob, current_sample - window_shift);
#endif  //__DEBUG_SPEECH_PROB___
        if (triggered == true) {
          // 语音分片后第一次遇到静音片，记录可能结束点
          //(The first silent segment after voice segmentation, recording
          // possible end point)
          if (temp_end == 0) {
            temp_end = current_sample;
          }
          // 大于累计静音最大值，记录可能结束点，用于大语音片强制分片时用
          // （If it is greater than the maximum value of accumulated silence,
          // the possible end point is recorded and used for forced segmentation
          // of large audio clips.）
          if (current_sample - temp_end > min_silence_samples_at_max_speech)
            prev_end = temp_end;
          // a. silence < min_slience_samples, continue speaking
          if ((current_sample - temp_end) < min_silence_samples) {
          }
          // b. silence >= min_slience_samples, end speaking
          else {
            current_speech.end = temp_end;
            if (current_speech.end - current_speech.start >
                min_speech_samples) {
#ifdef __DEBUG_SPEECH_PROB___
              printf("{>min speech start: %d, end:%d}\n", current_speech.start,
                     current_speech.end);
#endif  //__DEBUG_SPEECH_PROB___
              std::vector<float> s =
                  buffer_.Get(current_speech.start,
                              current_speech.end - current_speech.start);
              SpeechSegment segment;
              segment.start = current_speech.start;
              segment.samples = std::move(s);
              segments_.push(std::move(segment));
              current_speech = timestamp_t();
              prev_end = 0;
              next_start = 0;
              temp_end = 0;
              triggered = false;
            }
          }
        } else {
          // may first windows see end state.
        }
        continue;
      }
    }
    last_ = std::vector<float>(
        p, static_cast<const float *>(last_.data()) + last_.size());

    if (current_speech.start > 0) {
      buffer_.Pop(current_speech.start - buffer_.Head());
    } else {
      buffer_.Pop(current_sample - buffer_.Head());
    }
  }

  bool Empty() const { return segments_.empty(); }

  void Pop() { segments_.pop(); }

  void Clear() { std::queue<SpeechSegment>().swap(segments_); }

  const SpeechSegment &Front() const { return segments_.front(); }

  void Reset() {
    std::queue<SpeechSegment>().swap(segments_);
    model_->Reset();
    buffer_.Reset();
    // 重置相关变量
    current_sample = 0;
    current_speech = timestamp_t();
    prev_end = 0;
    next_start = 0;
    temp_end = 0;
    triggered = false;
    last_.clear();
  }

  void Flush() {
    int32_t buffer_size = buffer_.Size();

    if (buffer_size > 0) {
      std::vector<float> s = buffer_.Get(buffer_.Head(), buffer_size);
      SpeechSegment segment;
      segment.start = current_sample;
      segment.samples = std::move(s);
      segments_.push(std::move(segment));
      buffer_.Pop(buffer_size);
    }
  }

  bool IsSpeechDetected() const { return !segments_.empty(); }

  const VadModelConfig &GetConfig() const { return config_; }

 private:
  std::queue<SpeechSegment> segments_;
  timestamp_t current_speech;

  std::unique_ptr<SileroVadModel> model_;
  VadModelConfig config_;
  CircularBuffer buffer_;
  std::vector<float> last_;
  int32_t window_size;
  int32_t window_shift;
  int32_t sample_rate;
  float threshold;
  int32_t min_silence_samples;                // sr_per_ms * #ms
  int32_t min_silence_samples_at_max_speech;  // sr_per_ms * #98
  int32_t min_speech_samples;                 // sr_per_ms * #ms
  int32_t max_speech_samples;
  int32_t speech_pad_samples;  // usually a

  // model states
  bool triggered = false;
  unsigned int temp_end = 0;
  unsigned int current_sample = 0;
  // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes
  int32_t prev_end = 0;
  int32_t next_start = 0;
};

VoiceActivityDetector::VoiceActivityDetector(
    const VadModelConfig &config, float buffer_size_in_seconds /*= 60*/)
    : impl_(std::make_unique<Impl>(config, buffer_size_in_seconds)) {}

#if __ANDROID_API__ >= 9
VoiceActivityDetector::VoiceActivityDetector(
    AAssetManager *mgr, const VadModelConfig &config,
    float buffer_size_in_seconds /*= 60*/)
    : impl_(std::make_unique<Impl>(mgr, config, buffer_size_in_seconds)) {}
#endif

VoiceActivityDetector::~VoiceActivityDetector() = default;

void VoiceActivityDetector::AcceptWaveform(const float *samples, int32_t n) {
  impl_->AcceptWaveform(samples, n);
}

bool VoiceActivityDetector::Empty() const { return impl_->Empty(); }

void VoiceActivityDetector::Pop() { impl_->Pop(); }

void VoiceActivityDetector::Clear() { impl_->Clear(); }

const SpeechSegment &VoiceActivityDetector::Front() const {
  return impl_->Front();
}

void VoiceActivityDetector::Reset() const { impl_->Reset(); }

void VoiceActivityDetector::Flush() const { impl_->Flush(); }

bool VoiceActivityDetector::IsSpeechDetected() const {
  return impl_->IsSpeechDetected();
}

const VadModelConfig &VoiceActivityDetector::GetConfig() const {
  return impl_->GetConfig();
}

}  // namespace sherpa_onnx
