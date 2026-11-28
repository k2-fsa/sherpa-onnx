// sherpa-onnx/csrc/offline-tts-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
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

// 'validate' checks the selected model config either against the file
// system or against a resource manager.
template <typename ValidateModel>
static bool ValidateOfflineTtsModelConfig(const OfflineTtsModelConfig &config,
                                          const ValidateModel &validate) {
  if (config.num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", config.num_threads);
    return false;
  }

  if (!config.vits.model.empty()) {
    return validate(config.vits);
  }

  if (!config.matcha.acoustic_model.empty()) {
    return validate(config.matcha);
  }

  if (!config.zipvoice.decoder.empty()) {
    return validate(config.zipvoice);
  }

  if (!config.kokoro.model.empty()) {
    return validate(config.kokoro);
  }

  if (!config.kitten.model.empty()) {
    return validate(config.kitten);
  }

  if (!config.pocket.lm_flow.empty()) {
    return validate(config.pocket);
  }

  if (!config.supertonic.tts_json.empty()) {
    return validate(config.supertonic);
  }

  SHERPA_ONNX_LOGE("Please provide exactly one tts model.");

  return false;
}

bool OfflineTtsModelConfig::Validate() const {
  return ValidateOfflineTtsModelConfig(
      *this, [](const auto &c) -> bool { return c.Validate(); });
}

template <typename Manager>
bool OfflineTtsModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineTtsModelConfig(
      *this, [mgr](const auto &c) -> bool { return c.Validate(mgr); });
}

#if __ANDROID_API__ >= 9
template bool OfflineTtsModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineTtsModelConfig::Validate(NativeResourceManager *mgr) const;
#endif

std::string OfflineTtsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsModelConfig(";
  os << "vits=" << vits.ToString() << ", ";
  os << "matcha=" << matcha.ToString() << ", ";
  os << "kokoro=" << kokoro.ToString() << ", ";
  os << "zipvoice=" << zipvoice.ToString() << ", ";
  os << "kitten=" << kitten.ToString() << ", ";
  os << "pocket=" << pocket.ToString() << ", ";
  os << "supertonic=" << supertonic.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
