// sherpa-onnx/csrc/online-speech-denoiser.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_H_
#define SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_H_

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

class OnlineSpeechDenoiserImpl;

struct OnlineSpeechDenoiserConfig {
  OfflineSpeechDenoiserModelConfig model;

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class OnlineSpeechDenoiser {
 public:
  explicit OnlineSpeechDenoiser(const OnlineSpeechDenoiserConfig &config);
  ~OnlineSpeechDenoiser();

  template <typename Manager>
  OnlineSpeechDenoiser(Manager *mgr, const OnlineSpeechDenoiserConfig &config);

  /*
   * Process one chunk of streaming audio and return the enhanced samples
   * currently available. Internally this keeps model and overlap-add state
   * across calls.
   */
  DenoisedAudio Run(const float *samples, int32_t n, int32_t sample_rate);

  /*
   * Flush any buffered audio and reset the denoiser to an empty state so it
   * can be reused for a new stream.
   */
  DenoisedAudio Flush();

  void Reset();

  int32_t GetSampleRate() const;
  int32_t GetFrameShiftInSamples() const;

 private:
  std::unique_ptr<OnlineSpeechDenoiserImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_SPEECH_DENOISER_H_
