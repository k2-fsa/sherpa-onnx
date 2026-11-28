// sherpa-onnx/csrc/online-vad-model.h
//
// Copyright (c)  2026  Manickavela Arumugam
//
// Abstract base class for streaming (online) VAD models.
//
// A model owns its read-only session, built once per recognizer, and keeps
// all mutable state on the OnlineStream it scores. It is driven by
// OnlineRecognizer::VadProcess(). When a stream's speech -> silence flip is
// confirmed it calls OnlineStream::VadPushPrediction(), which arms the
// stream's endpoint and end-of-utterance flush.

#ifndef SHERPA_ONNX_CSRC_ONLINE_VAD_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_VAD_MODEL_H_

#include <cstdint>
#include <memory>

#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/online-vad-model-config.h"

namespace sherpa_onnx {

class OnlineVadModel {
 public:
  virtual ~OnlineVadModel() = default;

  // Returns nullptr if the config is disabled.
  static std::unique_ptr<OnlineVadModel> Create(
      const OnlineVadModelConfig &config);

  // Scores every full window of pending audio on ss[0..n), batched.
  virtual void RunVad(OnlineStream **ss, int32_t n) = 0;

  // Runs `n` dummy inferences.
  virtual void WarmUp(int32_t n) = 0;

  // Short tag, e.g. "silero".
  virtual const char *Name() const = 0;

  // Sample rate (Hz) of the audio this model scores.
  virtual int32_t SampleRate() const = 0;

  // Stride between windows, in samples.
  virtual int32_t WindowShift() const = 0;

  // Samples per inference (stride + context).
  virtual int32_t WindowSize() const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_VAD_MODEL_H_
