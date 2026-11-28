// sherpa-onnx/csrc/offline-speaker-segmentation-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-onnx/csrc/offline-speaker-segmentation-model-config.h"

#include <sstream>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSpeakerSegmentationModelConfig::Register(ParseOptions *po) {
  pyannote.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");

  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml");
}

// 'validate_pyannote' checks the pyannote config either against the file
// system or against a resource manager.
template <typename ValidatePyannote>
static bool ValidateOfflineSpeakerSegmentationModelConfig(
    const OfflineSpeakerSegmentationModelConfig &config,
    const ValidatePyannote &validate_pyannote) {
  if (config.num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d",
                     config.num_threads);
    return false;
  }

  if (!config.pyannote.model.empty()) {
    return validate_pyannote();
  }

  if (config.pyannote.model.empty()) {
    SHERPA_ONNX_LOGE(
        "You have to provide at least one speaker segmentation model");
    return false;
  }

  return true;
}

bool OfflineSpeakerSegmentationModelConfig::Validate() const {
  return ValidateOfflineSpeakerSegmentationModelConfig(
      *this, [this]() -> bool { return pyannote.Validate(); });
}

template <typename Manager>
bool OfflineSpeakerSegmentationModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineSpeakerSegmentationModelConfig(
      *this, [this, mgr]() -> bool { return pyannote.Validate(mgr); });
}

#if __ANDROID_API__ >= 9
template bool OfflineSpeakerSegmentationModelConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineSpeakerSegmentationModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineSpeakerSegmentationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeakerSegmentationModelConfig(";
  os << "pyannote=" << pyannote.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
