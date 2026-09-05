// sherpa-onnx/csrc/online-speech-denoiser.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-speech-denoiser.h"

#include <memory>
#include <sstream>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/online-speech-denoiser-impl.h"

namespace sherpa_onnx {

void OnlineSpeechDenoiserConfig::Register(ParseOptions *po) {
  model.Register(po);
}

bool OnlineSpeechDenoiserConfig::Validate() const { return model.Validate(); }

std::string OnlineSpeechDenoiserConfig::ToString() const {
  std::ostringstream os;
  os << "OnlineSpeechDenoiserConfig(";
  os << "model=" << model.ToString() << ")";
  return os.str();
}

template <typename Manager>
OnlineSpeechDenoiser::OnlineSpeechDenoiser(
    Manager *mgr, const OnlineSpeechDenoiserConfig &config)
    : impl_(OnlineSpeechDenoiserImpl::Create(mgr, config)) {}

OnlineSpeechDenoiser::OnlineSpeechDenoiser(
    const OnlineSpeechDenoiserConfig &config)
    : impl_(OnlineSpeechDenoiserImpl::Create(config)) {}

OnlineSpeechDenoiser::~OnlineSpeechDenoiser() = default;

DenoisedAudio OnlineSpeechDenoiser::Run(const float *samples, int32_t n,
                                        int32_t sample_rate) {
  return impl_->Run(samples, n, sample_rate);
}

DenoisedAudio OnlineSpeechDenoiser::Flush() { return impl_->Flush(); }

void OnlineSpeechDenoiser::Reset() { impl_->Reset(); }

int32_t OnlineSpeechDenoiser::GetSampleRate() const {
  return impl_->GetSampleRate();
}

int32_t OnlineSpeechDenoiser::GetFrameShiftInSamples() const {
  return impl_->GetFrameShiftInSamples();
}

#if __ANDROID_API__ >= 9
template OnlineSpeechDenoiser::OnlineSpeechDenoiser(
    AAssetManager *mgr, const OnlineSpeechDenoiserConfig &config);
#endif

#if __OHOS__
template OnlineSpeechDenoiser::OnlineSpeechDenoiser(
    NativeResourceManager *mgr, const OnlineSpeechDenoiserConfig &config);
#endif

}  // namespace sherpa_onnx
