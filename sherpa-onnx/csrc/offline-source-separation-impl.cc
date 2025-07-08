// sherpa-onnx/csrc/offline-source-separation-impl.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-source-separation-impl.h"

#include <algorithm>
#include <memory>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/offline-source-separation-spleeter-impl.h"
#include "sherpa-onnx/csrc/offline-source-separation-uvr-impl.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineSourceSeparationImpl>
OfflineSourceSeparationImpl::Create(
    const OfflineSourceSeparationConfig &config) {
  if (!config.model.spleeter.vocals.empty()) {
    return std::make_unique<OfflineSourceSeparationSpleeterImpl>(config);
  }

  if (!config.model.uvr.model.empty()) {
    return std::make_unique<OfflineSourceSeparationUvrImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please provide a separation model!");

  return nullptr;
}

template <typename Manager>
std::unique_ptr<OfflineSourceSeparationImpl>
OfflineSourceSeparationImpl::Create(
    Manager *mgr, const OfflineSourceSeparationConfig &config) {
  if (!config.model.spleeter.vocals.empty()) {
    return std::make_unique<OfflineSourceSeparationSpleeterImpl>(mgr, config);
  }

  if (!config.model.uvr.model.empty()) {
    return std::make_unique<OfflineSourceSeparationUvrImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please provide a separation model!");

  return nullptr;
}

OfflineSourceSeparationInput OfflineSourceSeparationImpl::Resample(
    const OfflineSourceSeparationInput &input, bool debug /*= false*/) const {
  const OfflineSourceSeparationInput *p_input = &input;
  OfflineSourceSeparationInput tmp_input;

  int32_t output_sample_rate = GetOutputSampleRate();

  if (input.sample_rate != output_sample_rate) {
    SHERPA_ONNX_LOGE(
        "Creating a resampler:\n"
        "   in_sample_rate: %d\n"
        "   output_sample_rate: %d\n",
        input.sample_rate, output_sample_rate);

    float min_freq = std::min<int32_t>(input.sample_rate, output_sample_rate);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    auto resampler =
        std::make_unique<LinearResample>(input.sample_rate, output_sample_rate,
                                         lowpass_cutoff, lowpass_filter_width);

    std::vector<float> s;
    for (const auto &samples : input.samples.data) {
      resampler->Reset();
      resampler->Resample(samples.data(), samples.size(), true, &s);
      tmp_input.samples.data.push_back(std::move(s));
    }

    tmp_input.sample_rate = output_sample_rate;
    p_input = &tmp_input;
  }

  if (p_input->samples.data.size() > 1) {
    if (debug) {
      SHERPA_ONNX_LOGE("input ch1 samples size: %d",
                       static_cast<int32_t>(p_input->samples.data[1].size()));
    }

    if (p_input->samples.data[0].size() != p_input->samples.data[1].size()) {
      SHERPA_ONNX_LOGE("ch0 samples size %d vs ch1 samples size %d",
                       static_cast<int32_t>(p_input->samples.data[0].size()),
                       static_cast<int32_t>(p_input->samples.data[1].size()));

      SHERPA_ONNX_EXIT(-1);
    }
  }

  return *p_input;
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineSourceSeparationImpl>
OfflineSourceSeparationImpl::Create(
    AAssetManager *mgr, const OfflineSourceSeparationConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineSourceSeparationImpl>
OfflineSourceSeparationImpl::Create(
    NativeResourceManager *mgr, const OfflineSourceSeparationConfig &config);
#endif

}  // namespace sherpa_onnx
