// sherpa-onnx/csrc/offline-speech-denoiser-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speech-denoiser-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSpeechDenoiserModelConfig::Register(ParseOptions *po) {
  gtcrn.Register(po);
  dpdfnet.Register(po);
  voxlock.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

// Checks that a model is given. Shared by Validate() and Validate(Manager *).
static bool HasSpeechDenoiserModel(
    const OfflineSpeechDenoiserModelConfig &config) {
  const bool has_voxlock = !config.voxlock.enroll_model.empty() ||
                           !config.voxlock.chunk_model.empty();

  if (config.gtcrn.model.empty() && config.dpdfnet.model.empty() &&
      !has_voxlock) {
    SHERPA_ONNX_LOGE("Please provide a speech denoising model.");
    return false;
  }

  return true;
}

bool OfflineSpeechDenoiserModelConfig::Validate() const {
  if (!HasSpeechDenoiserModel(*this)) {
    return false;
  }

  if (!gtcrn.model.empty()) {
    return gtcrn.Validate();
  }

  if (!dpdfnet.model.empty()) {
    return dpdfnet.Validate();
  }

  return voxlock.Validate();
}

template <typename Manager>
bool OfflineSpeechDenoiserModelConfig::Validate(Manager *mgr) const {
  if (!HasSpeechDenoiserModel(*this)) {
    return false;
  }

  if (!gtcrn.model.empty()) {
    return gtcrn.Validate(mgr);
  }

  if (!dpdfnet.model.empty()) {
    return dpdfnet.Validate(mgr);
  }

  return voxlock.Validate(mgr);
}

#if __ANDROID_API__ >= 9
template bool OfflineSpeechDenoiserModelConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineSpeechDenoiserModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineSpeechDenoiserModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserModelConfig(";
  os << "gtcrn=" << gtcrn.ToString() << ", ";
  os << "dpdfnet=" << dpdfnet.ToString() << ", ";
  os << "voxlock=" << voxlock.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
