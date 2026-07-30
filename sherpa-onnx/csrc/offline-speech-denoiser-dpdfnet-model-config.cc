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

bool OfflineSpeechDenoiserDpdfNetModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --speech-denoiser-dpdfnet-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("dpdfnet model file '%s' does not exist", model.c_str());
    return false;
  }

  if (std::isnan(attenuation_limit_db) || attenuation_limit_db < 0.0f ||
      (attenuation_limit_db > 100.0f && !std::isinf(attenuation_limit_db))) {
    SHERPA_ONNX_LOGE(
        "attenuation_limit_db must be in [0, 100] or infinity. Given: %f",
        attenuation_limit_db);
    return false;
  }

  return true;
}

std::string OfflineSpeechDenoiserDpdfNetModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserDpdfNetModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "attenuation_limit_db=" << attenuation_limit_db << ")";
  return os.str();
}

}  // namespace sherpa_onnx
