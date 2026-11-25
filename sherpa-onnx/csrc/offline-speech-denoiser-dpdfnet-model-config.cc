// sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model-config.cc
//
// Copyright (c)  2026  Ceva Inc

#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model-config.h"

#include <cmath>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSpeechDenoiserDpdfNetModelConfig::Register(ParseOptions *po) {
  po->Register("speech-denoiser-dpdfnet-model", &model,
               "Path to a DPDFNet ONNX model for speech denoising, e.g. "
               "baseline/dpdfnet2/dpdfnet4/dpdfnet8 (16 kHz), "
               "dpdfnet2_8khz/dpdfnet8_8khz (8 kHz), or "
               "dpdfnet2_48khz_hr/dpdfnet8_48khz_hr (48 kHz). "
               "Download DPDFNet models from the "
               "sherpa-onnx GitHub release or the official Hugging Face hub: "
               "https://github.com/k2-fsa/sherpa-onnx/releases/tag/"
               "speech-enhancement-models or "
               "https://huggingface.co/Ceva-IP/DPDFNet");

  po->Register(
      "speech-denoiser-dpdfnet-attenuation-limit-db", &attenuation_limit_db,
      "Offline-only DPDFNet attenuation limit in dB. Values greater than 0 "
      "limit suppression by blending aligned noisy spectra into the enhanced "
      "spectra. The maximum finite value is 100. 0 or infinity disables the "
      "limit.");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateDpdfNetModelConfig(
    const OfflineSpeechDenoiserDpdfNetModelConfig &config,
    const Exists &exists) {
  if (config.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --speech-denoiser-dpdfnet-model");
    return false;
  }

  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("dpdfnet model file '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  if (std::isnan(config.attenuation_limit_db) ||
      config.attenuation_limit_db < 0.0f ||
      (config.attenuation_limit_db > 100.0f &&
       !std::isinf(config.attenuation_limit_db))) {
    SHERPA_ONNX_LOGE(
        "attenuation_limit_db must be in [0, 100] or infinity. Given: %f",
        config.attenuation_limit_db);
    return false;
  }

  return true;
}

bool OfflineSpeechDenoiserDpdfNetModelConfig::Validate() const {
  return ValidateDpdfNetModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineSpeechDenoiserDpdfNetModelConfig::Validate(Manager *mgr) const {
  return ValidateDpdfNetModelConfig(*this, [mgr](const std::string &p) -> bool {
    return FileExists(mgr, p);
  });
}

#if __ANDROID_API__ >= 9
template bool OfflineSpeechDenoiserDpdfNetModelConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineSpeechDenoiserDpdfNetModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineSpeechDenoiserDpdfNetModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserDpdfNetModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "attenuation_limit_db=" << attenuation_limit_db << ")";
  return os.str();
}

}  // namespace sherpa_onnx
