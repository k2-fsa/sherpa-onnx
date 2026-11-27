// sherpa-onnx/csrc/offline-diacritization-model-config.cc
//
// Copyright (c)  2026  Matias Lin

#include "sherpa-onnx/csrc/offline-diacritization-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineDiacritizationModelConfig::Register(ParseOptions *po) {
  po->Register("catt-encoder", &catt_encoder,
               "Path to the CATT (Context-Aware Transformer for "
               "Tashkeel) ONNX encoder model for Arabic diacritization.");

  po->Register("catt-decoder", &catt_decoder,
               "Path to the CATT (Context-Aware Transformer for "
               "Tashkeel) ONNX decoder model for Arabic diacritization.");

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
static bool ValidateOfflineDiacritizationModelConfig(
    const OfflineDiacritizationModelConfig &config, const Exists &exists) {
  if (config.catt_encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --catt-encoder");
    return false;
  }

  if (config.catt_decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --catt-decoder");
    return false;
  }

  if (!exists(config.catt_encoder)) {
    SHERPA_ONNX_LOGE("--catt-encoder %s does not exist",
                     config.catt_encoder.c_str());
    return false;
  }

  if (!exists(config.catt_decoder)) {
    SHERPA_ONNX_LOGE("--catt-decoder %s does not exist",
                     config.catt_decoder.c_str());
    return false;
  }

  return true;
}

bool OfflineDiacritizationModelConfig::Validate() const {
  return ValidateOfflineDiacritizationModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineDiacritizationModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineDiacritizationModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OfflineDiacritizationModelConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineDiacritizationModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineDiacritizationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineDiacritizationModelConfig(";
  os << "catt_encoder=\"" << catt_encoder << "\", ";
  os << "catt_decoder=\"" << catt_decoder << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
