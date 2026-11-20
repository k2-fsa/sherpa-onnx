// sherpa-onnx/csrc/vad-stream-state.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VAD_STREAM_STATE_H_
#define SHERPA_ONNX_CSRC_VAD_STREAM_STATE_H_

#include <memory>
#include <queue>
#include <vector>

#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/vad-model-config.h"
#include "sherpa-onnx/csrc/vad-model.h"

namespace knf {
class LinearResample;
}  // namespace knf

namespace sherpa_onnx {

struct SpeechSegment {
  // Start of this segment in the input stream, in samples.
  //
  // It is an absolute position and keeps increasing during the lifetime of
  // the detector, so it is 64-bit. 2^63 samples is about 18 million years
  // at 16 kHz.
  //
  // It is -1 if this segment is not valid.
  int64_t start = -1;
  std::vector<float> samples;
};

// The per-stream state of a VoiceActivityDetector.
//
// A VoiceActivityDetector shares its model among the streams it creates,
// so everything that changes as windows are processed lives here: the
// samples waiting to be processed, the model state saved from the last
// window, and the speech segments found so far. Each stream is processed
// independently, so one detector can serve several audio inputs.
class VadStreamState {
 public:
  VadStreamState(const VadModelConfig& config, float buffer_size_in_seconds);
  ~VadStreamState();

  // The VAD sample rate the accepted samples are converted to.
  int32_t ExpectedSampleRate() const { return sample_rate_; }

  // Resample `waveform` to the VAD sample rate if needed and buffer it.
  //
  // @param sampling_rate The sampling_rate of the input waveform. If it
  //                      does not equal the VAD sample rate, we do
  //                      resampling inside.
  // @param waveform Pointer to a 1-D array of size n. It must be
  //                 normalized to the range [-1, 1].
  // @param n Number of entries in waveform
  void AcceptWaveform(int32_t sampling_rate, const float* waveform, int32_t n);

  // Return true if the number of samples in the stream is >= the model's
  // window size. Return false otherwise.
  bool IsReady(const VadModel* model) const;

  // Process one window of samples with `model`. If IsReady() returns
  // false, do nothing.
  void Process(VadModel* model);

  // Process any remaining windows and save the speech segment that is
  // still open, if any.
  void Flush(VadModel* model);

  void Reset(VadModel* model);

  bool Empty() const { return segments_.empty(); }

  void Pop() { segments_.pop(); }

  void Clear() { std::queue<SpeechSegment>().swap(segments_); }

  // It is an error to call Front() if Empty() returns true.
  //
  // The returned reference is valid until the next call to any methods
  // of this class.
  const SpeechSegment& Front() const;

  bool IsSpeechDetected() const { return start_ != -1; }

  // It is empty if IsSpeechDetected() returns false
  SpeechSegment CurrentSpeechSegment() const { return cur_segment_; }

 private:
  // The samples received so far and not processed yet. They are shorter
  // than a window when IsReady() returns false.
  std::vector<float> samples_;

  std::unique_ptr<knf::LinearResample> resampler_;

  VadModelState model_state_;

  std::queue<SpeechSegment> segments_;

  // it is empty if no speech is detected
  SpeechSegment cur_segment_;

  CircularBuffer buffer_;

  VadModelConfig config_;

  int32_t sample_rate_ = 16000;
  int32_t max_utterance_length_ = -1;  // in samples

  int64_t start_ = -1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VAD_STREAM_STATE_H_
