// sherpa-onnx/csrc/online-vad-model-config.h
//
// Copyright (c)  2026  Manickavela Arumugam

#ifndef SHERPA_ONNX_CSRC_ONLINE_VAD_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_VAD_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

// Configuration for an independent Silero VAD pipeline that runs
// *parallel to* the online (streaming) recognizer's decode loop --
// mirroring ds-k2-sherpa-onnx's refactor_vad architecture: a VadProcess()
// step on its own schedule, reading an independent copy of each stream's
// raw audio, writing predictions to a per-stream OnlineVadBuffer that
// OnlineRecognizer::IsReady()'s coverage gate and IsEndpoint() read.
//
// Disabled when `model` is empty -- zero overhead and zero behavior
// change for callers who don't set it. Unlike the inline-VAD branch of
// this port, there is no "run inline inside AcceptWaveform" fallback:
// VadProcess() must be driven explicitly (the websocket server does
// this on its own io_vad_ timer), exactly as in the source architecture.
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
