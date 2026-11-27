// sherpa-onnx/csrc/audio-tagging-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void AudioTaggingModelConfig::Register(ParseOptions *po) {
  zipformer.Register(po);

  po->Register("ced-model", &ced,
               "Path to CED model. Only need to pass one of --zipformer-model "
               "or --ced-model");

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

// 'validate_zipformer' and 'exists' check the zipformer config and resolve
// a path either against the file system or against a resource manager.
template <typename ValidateZipformer, typename Exists>
static bool ValidateAudioTaggingModelConfig(
    const AudioTaggingModelConfig &config,
    const ValidateZipformer &validate_zipformer, const Exists &exists) {
  if (!config.zipformer.model.empty() && !validate_zipformer()) {
    return false;
  }

  if (!config.ced.empty() && !exists(config.ced)) {
    SHERPA_ONNX_LOGE("CED model file '%s' does not exist", config.ced.c_str());
    return false;
  }

  if (config.zipformer.model.empty() && config.ced.empty()) {
    SHERPA_ONNX_LOGE("Please provide either --zipformer-model or --ced-model");
    return false;
  }

  return true;
}

bool AudioTaggingModelConfig::Validate() const {
  return ValidateAudioTaggingModelConfig(
      *this, [this]() -> bool { return zipformer.Validate(); },
      [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool AudioTaggingModelConfig::Validate(Manager *mgr) const {
  return ValidateAudioTaggingModelConfig(
      *this, [this, mgr]() -> bool { return zipformer.Validate(mgr); },
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool AudioTaggingModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool AudioTaggingModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string AudioTaggingModelConfig::ToString() const {
  std::ostringstream os;

  os << "AudioTaggingModelConfig(";
  os << "zipformer=" << zipformer.ToString() << ", ";
  os << "ced=\"" << ced << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
