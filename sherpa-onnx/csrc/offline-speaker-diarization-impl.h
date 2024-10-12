// sherpa-onnx/csrc/offline-speaker-diarization-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_IMPL_H_

#include <functional>
#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/offline-speaker-diarization.h"
namespace sherpa_onnx {

class OfflineSpeakerDiarizationImpl {
 public:
  static std::unique_ptr<OfflineSpeakerDiarizationImpl> Create(
      const OfflineSpeakerDiarizationConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<OfflineSpeakerDiarizationImpl> Create(
      AAssetManager *mgr, const OfflineSpeakerDiarizationConfig &config);
#endif

  virtual ~OfflineSpeakerDiarizationImpl() = default;

  virtual int32_t SampleRate() const = 0;

  // Note: Only config.clustering is used. All other fields in config are
  // ignored
  virtual void SetConfig(const OfflineSpeakerDiarizationConfig &config) = 0;

  virtual OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_IMPL_H_
