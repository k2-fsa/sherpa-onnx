// sherpa-onnx/csrc/offline-speaker-diarization-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_IMPL_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-speaker-diarization.h"
namespace sherpa_onnx {

class OfflineSpeakerDiarizationImpl {
 public:
  static std::unique_ptr<OfflineSpeakerDiarizationImpl> Create(
      const OfflineSpeakerDiarizationConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineSpeakerDiarizationImpl> Create(
      Manager *mgr, const OfflineSpeakerDiarizationConfig &config);

  virtual ~OfflineSpeakerDiarizationImpl() = default;

  virtual int32_t SampleRate() const = 0;

  // Note: Only config.clustering is used. All other fields in config are
  // ignored
  virtual void SetConfig(const OfflineSpeakerDiarizationConfig &config) = 0;

  virtual OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const = 0;

  virtual OfflineSpeakerDiarizationResult Process(
      const OfflineStream *stream,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const {
    std::vector<float> samples = stream->GetFrames();
    return Process(samples.data(), static_cast<int32_t>(samples.size()),
                   std::move(callback), callback_arg);
  }
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_IMPL_H_
