// sherpa-onnx/csrc/offline-speech-denoiser.h
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speech-denoiser.h"

#include <cmath>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser-impl.h"

namespace sherpa_onnx {

void OfflineSpeechDenoiserConfig::Register(ParseOptions *po) {
  model.Register(po);
  po->Register(
      "speech-denoiser-dpdfnet-attenuation-limit-db",
      &dpdfnet_attenuation_limit_db,
      "Offline-only DPDFNet attenuation limit in dB. Values greater than 0 "
      "limit suppression by blending aligned noisy spectra into the enhanced "
      "spectra. 0 or infinity disables the limit.");
}

bool OfflineSpeechDenoiserConfig::Validate() const {
  if (std::isnan(dpdfnet_attenuation_limit_db) ||
      dpdfnet_attenuation_limit_db < 0.0f) {
    SHERPA_ONNX_LOGE(
        "dpdfnet_attenuation_limit_db must be non-negative. Given: %f",
        dpdfnet_attenuation_limit_db);
    return false;
  }

  if (dpdfnet_attenuation_limit_db > 0.0f &&
      !std::isinf(dpdfnet_attenuation_limit_db) &&
      model.dpdfnet.model.empty()) {
    SHERPA_ONNX_LOGE(
        "dpdfnet_attenuation_limit_db is supported only with a DPDFNet model");
    return false;
  }

  return model.Validate();
}

std::string OfflineSpeechDenoiserConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeechDenoiserConfig(";
  os << "model=" << model.ToString() << ", ";
  os << "dpdfnet_attenuation_limit_db="
     << dpdfnet_attenuation_limit_db << ")";
  return os.str();
}

template <typename Manager>
OfflineSpeechDenoiser::OfflineSpeechDenoiser(
    Manager *mgr, const OfflineSpeechDenoiserConfig &config)
    : impl_(OfflineSpeechDenoiserImpl::Create(mgr, config)) {}

OfflineSpeechDenoiser::OfflineSpeechDenoiser(
    const OfflineSpeechDenoiserConfig &config)
    : impl_(OfflineSpeechDenoiserImpl::Create(config)) {}

OfflineSpeechDenoiser::~OfflineSpeechDenoiser() = default;

DenoisedAudio OfflineSpeechDenoiser::Run(const float *samples, int32_t n,
                                         int32_t sample_rate) const {
  return impl_->Run(samples, n, sample_rate);
}

int32_t OfflineSpeechDenoiser::GetSampleRate() const {
  return impl_->GetSampleRate();
}

#if __ANDROID_API__ >= 9
template OfflineSpeechDenoiser::OfflineSpeechDenoiser(
    AAssetManager *mgr, const OfflineSpeechDenoiserConfig &config);
#endif

#if __OHOS__
template OfflineSpeechDenoiser::OfflineSpeechDenoiser(
    NativeResourceManager *mgr, const OfflineSpeechDenoiserConfig &config);
#endif

}  // namespace sherpa_onnx
