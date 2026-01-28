// sherpa-onnx/csrc/offline-tts-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsModelConfig::Register(ParseOptions *po) {
  vits.Register(po);
  matcha.Register(po);
  kokoro.Register(po);
  zipvoice.Register(po);
  kitten.Register(po);
  pocket.Register(po);
  supertonic.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

bool OfflineTtsModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (!vits.model.empty()) {
    return vits.Validate();
  }

  if (!matcha.acoustic_model.empty()) {
    return matcha.Validate();
  }

  if (!zipvoice.decoder.empty()) {
    return zipvoice.Validate();
  }

  if (!kokoro.model.empty()) {
    return kokoro.Validate();
  }

  if (!kitten.model.empty()) {
    return kitten.Validate();
  }

  if (!pocket.lm_flow.empty()) {
    return pocket.Validate();
  }

  if (!supertonic.model_dir.empty()) {
    return supertonic.Validate();
  }

  SHERPA_ONNX_LOGE("Please provide exactly one tts model.");

  return false;
}

std::string OfflineTtsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsModelConfig(";
  os << "vits=" << vits.ToString() << ", ";
  os << "matcha=" << matcha.ToString() << ", ";
  os << "kokoro=" << kokoro.ToString() << ", ";
  os << "zipvoice=" << zipvoice.ToString() << ", ";
  os << "kitten=" << kitten.ToString() << ", ";
  os << "supertonic=" << supertonic.ToString() << ", ";
  os << "pocket=" << pocket.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
