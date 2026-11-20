// sherpa-onnx/csrc/offline-punctuation-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-punctuation-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflinePunctuationModelConfig::Register(ParseOptions *po) {
  po->Register("ct-transformer", &ct_transformer,
               "Path to the controllable time-delay (CT) transformer model");

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOfflinePunctuationModelConfig(
    const OfflinePunctuationModelConfig &config, const Exists &exists) {
  if (config.ct_transformer.empty()) {
    SHERPA_ONNX_LOGE("Please provide --ct-transformer");
    return false;
  }

  if (!exists(config.ct_transformer)) {
    SHERPA_ONNX_LOGE("--ct-transformer %s does not exist",
                     config.ct_transformer.c_str());
    return false;
  }

  return true;
}

bool OfflinePunctuationModelConfig::Validate() const {
  return ValidateOfflinePunctuationModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflinePunctuationModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflinePunctuationModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OfflinePunctuationModelConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflinePunctuationModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflinePunctuationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflinePunctuationModelConfig(";
  os << "ct_transformer=\"" << ct_transformer << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
