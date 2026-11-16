// sherpa-onnx/csrc/online-vad-model-config.h
//
// Copyright (c)  2026  Manickavela Arumugam

#ifndef SHERPA_ONNX_CSRC_ONLINE_VAD_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_VAD_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

// Configuration for the online Silero VAD. It runs next to the streaming
// recognizer: OnlineRecognizer::VadProcess() scores a per-stream copy of the
// audio, and OnlineRecognizer::IsReady() / IsEndpoint() read the result.
// Disabled when `model` is empty.
struct OnlineVadModelConfig {
  // Path to a Silero VAD ONNX model (v4 or v5). Empty disables this
  // feature entirely.
  std::string model;

  // Speech-probability threshold. See SileroVadModelConfig for details.
  float threshold = 0.5f;

  // Trailing silence (seconds) required before IsEndpoint() reports the
  // VAD-based rule.
  float min_silence_duration = 0.5f;

  // Sustained speech (seconds) required before the VAD confirms speech
  // has started.
  float min_speech_duration = 0.25f;

  // See SileroVadModelConfig::max_speech_duration.
  float max_speech_duration = 20.0f;

  // Window size in samples per Silero inference call. 16 kHz: 512, 1024,
  // 1536. 8 kHz: 256, 512, 768. (Silero v5 ONNX exports in this repo
  // require 512 regardless of sample rate -- see silero-vad-model.cc.)
  int32_t window_size = 512;

  // See SileroVadModelConfig::neg_threshold. < 0 means use the Silero
  // default rule.
  float neg_threshold = -1.0f;

  // Sample rate the Silero session is built for. Each stream's raw-audio
  // copy is resampled to this rate before being fed to the VAD,
  // independent of the ASR feature pipeline's sample rate.
  int32_t sample_rate = 16000;

  int32_t num_threads = 1;

  std::string provider = "cpu";

  bool debug = false;

  // If true, OnlineRecognizer::IsReady() fast-forwards num_processed_frames_
  // over any ASR chunk window the VAD has already confirmed is silence,
  // skipping ASR decode on it entirely (MaybeSkipVadSilence()). If false,
  // the VAD still drives endpointing as usual, but every chunk is decoded
  // regardless of what the VAD predicted. Default on.
  bool batch_skip = true;

  OnlineVadModelConfig() = default;

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;

  bool IsEnabled() const { return !model.empty(); }

  // Builds the VadModelConfig expected by VadModel::Create() /
  // VoiceActivityDetector.
  VadModelConfig ToVadModelConfig() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_VAD_MODEL_CONFIG_H_
