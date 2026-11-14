// sherpa-onnx/csrc/online-vad-model-config.cc
//
// Copyright (c)  2026  Manickavela Arumugam

#include "sherpa-onnx/csrc/online-vad-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineVadModelConfig::Register(ParseOptions *po) {
  po->Register("online-vad-model", &model,
               "Path to a Silero VAD ONNX model. If set, an independent "
               "VAD pipeline (VadProcess, on its own schedule, parallel "
               "to DecodeStreams) runs alongside ASR decoding and "
               "reports an additional endpoint rule driven by the VAD's "
               "own trailing-silence detection. Empty (the default) "
               "disables this feature.");

  po->Register("online-vad-threshold", &threshold,
               "Speech-probability threshold for the online VAD (0..1).");

  po->Register("online-vad-min-silence-duration", &min_silence_duration,
               "In seconds. Trailing silence the online VAD requires "
               "before considering the current speech segment finished.");

  po->Register("online-vad-min-speech-duration", &min_speech_duration,
               "In seconds. Sustained speech the online VAD requires "
               "before considering a new speech segment started.");

  po->Register("online-vad-max-speech-duration", &max_speech_duration,
               "In seconds. See --silero-vad-max-speech-duration.");

  po->Register("online-vad-window-size", &window_size,
               "In samples. See --silero-vad-window-size.");

  po->Register("online-vad-neg-threshold", &neg_threshold,
               "Negative (exit) threshold. If < 0, the Silero default rule "
               "applies.");

  po->Register("online-vad-sample-rate", &sample_rate,
               "Sample rate the online VAD session is built for. Each "
               "stream's raw-audio copy is resampled to this rate before "
               "being fed to the VAD.");

  po->Register("online-vad-num-threads", &num_threads,
               "Number of threads to run the online VAD model.");

  po->Register("online-vad-provider", &provider,
               "Specify a provider to run the online VAD model. Supported "
               "values: cpu, cuda, coreml, openvino");

  po->Register("online-vad-debug", &debug,
               "true to display debug information when loading the online "
               "VAD model");

  po->Register("online-vad-batch-skip", &batch_skip,
               "true (default) to skip ASR decoding entirely on chunks the "
               "online VAD has already confirmed are silence. false to let "
               "the VAD drive endpointing as usual but still decode every "
               "chunk.");
}

bool OnlineVadModelConfig::Validate() const {
  if (model.empty()) {
    // Disabled; nothing to validate.
    return true;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--online-vad-model: '%s' does not exist",
                     model.c_str());
    return false;
  }

  if (threshold <= 0.0f || threshold >= 1.0f) {
    SHERPA_ONNX_LOGE(
        "Invalid --online-vad-threshold=%f (must be in (0, 1))", threshold);
    return false;
  }

  if (min_silence_duration <= 0.0f) {
    SHERPA_ONNX_LOGE(
        "Invalid --online-vad-min-silence-duration=%f (must be > 0)",
        min_silence_duration);
    return false;
  }

  if (min_speech_duration <= 0.0f) {
    SHERPA_ONNX_LOGE(
        "Invalid --online-vad-min-speech-duration=%f (must be > 0)",
        min_speech_duration);
    return false;
  }

  if (max_speech_duration <= 0.0f) {
    SHERPA_ONNX_LOGE(
        "Invalid --online-vad-max-speech-duration=%f (must be > 0)",
        max_speech_duration);
    return false;
  }

  if (window_size <= 0) {
    SHERPA_ONNX_LOGE("Invalid --online-vad-window-size=%d (must be > 0)",
                     window_size);
    return false;
  }

  return true;
}

std::string OnlineVadModelConfig::ToString() const {
  std::ostringstream os;
  os << "OnlineVadModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "max_speech_duration=" << max_speech_duration << ", ";
  os << "window_size=" << window_size << ", ";
  os << "neg_threshold=" << neg_threshold << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "batch_skip=" << (batch_skip ? "True" : "False") << ")";
  return os.str();
}

VadModelConfig OnlineVadModelConfig::ToVadModelConfig() const {
  VadModelConfig vad_config;
  vad_config.silero_vad.model = model;
  vad_config.silero_vad.threshold = threshold;
  vad_config.silero_vad.min_silence_duration = min_silence_duration;
  vad_config.silero_vad.min_speech_duration = min_speech_duration;
  vad_config.silero_vad.max_speech_duration = max_speech_duration;
  vad_config.silero_vad.window_size = window_size;
  vad_config.silero_vad.neg_threshold = neg_threshold;
  vad_config.sample_rate = sample_rate;
  vad_config.num_threads = num_threads;
  vad_config.provider = provider;
  vad_config.debug = debug;
  return vad_config;
}

}  // namespace sherpa_onnx
