// sherpa-onnx/csrc/voice-activity-detector.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
#define SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/vad-model-config.h"
#include "sherpa-onnx/csrc/vad-stream-state.h"

namespace sherpa_onnx {

class OnlineStream;

class VoiceActivityDetector {
 public:
  explicit VoiceActivityDetector(const VadModelConfig &config,
                                 float buffer_size_in_seconds = 60);

  template <typename Manager>
  VoiceActivityDetector(Manager *mgr, const VadModelConfig &config,
                        float buffer_size_in_seconds = 60);

  ~VoiceActivityDetector();

  void AcceptWaveform(const float *samples, int32_t n);
  float Compute(const float *samples, int32_t n);

  // The VAD model is shared among the created streams, so several streams
  // can be processed with one model in memory. The VAD states of each
  // stream are kept inside the stream. Feed a stream with
  // stream->AcceptWaveform(), drive it with IsReady()/Process(), and read
  // its speech segments with the stream variants of Empty()/Front()/Pop()
  // below. Note that a detector processes either the samples of
  // AcceptWaveform() above or the streams it has created; do not mix the
  // two for one detector.
  //
  // Threading: different streams of one detector may be driven from
  // different threads. Process(), Flush(), and Reset() hold a
  // detector-level lock while they run, so the shared model handles one
  // window at a time and these calls are serialized across the streams
  // of a detector. Feeding a stream with AcceptWaveform() and the other
  // per-stream accessors take a per-stream lock, so one thread can feed
  // a stream while another thread processes it. No extra locking is
  // required from the caller.
  std::unique_ptr<OnlineStream> CreateStream() const;

  // Return true if the number of samples in the stream is >= window_size.
  // Return false otherwise.
  bool IsReady(const OnlineStream *stream) const;

  // Only process samples of a window size. If IsReady() returns false, do
  // nothing.
  void Process(const OnlineStream *stream) const;

  bool Empty() const;
  void Pop();
  void Clear();

  // It is an error to call Front() if Empty() returns true.
  //
  // The returned reference is valid until the next call to any
  // methods of VoiceActivityDetector.
  const SpeechSegment &Front() const;

  bool IsSpeechDetected() const;

  // It is empty if IsSpeechDetected() returns false
  SpeechSegment CurrentSpeechSegment() const;

  void Reset() const;

  // At the end of the utterance, you can invoke this method so that
  // the last speech segment can be detected.
  void Flush() const;

  const VadModelConfig &GetConfig() const;

  // The following methods operate on a stream created with CreateStream()
  // and mirror the corresponding methods above.

  bool Empty(const OnlineStream *stream) const;
  void Pop(const OnlineStream *stream);
  void Clear(const OnlineStream *stream) const;

  // It is an error to call Front() if Empty(stream) returns true.
  const SpeechSegment &Front(const OnlineStream *stream) const;

  bool IsSpeechDetected(const OnlineStream *stream) const;

  SpeechSegment CurrentSpeechSegment(const OnlineStream *stream) const;

  void Reset(const OnlineStream *stream) const;

  // At the end of the utterance, you can invoke this method so that the
  // last speech segment of the stream can be detected.
  void Flush(const OnlineStream *stream) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
