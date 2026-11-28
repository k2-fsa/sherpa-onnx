// sherpa-onnx/csrc/offline-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speaker-diarization.h"

#include <memory>
#include <string>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/offline-speaker-diarization-impl.h"

namespace sherpa_onnx {

void OfflineSpeakerDiarizationConfig::Register(ParseOptions *po) {
  ParseOptions po_segmentation("segmentation", po);
  segmentation.Register(&po_segmentation);

  ParseOptions po_embedding("embedding", po);
  embedding.Register(&po_embedding);

  ParseOptions po_clustering("clustering", po);
  clustering.Register(&po_clustering);

  po->Register("min-duration-on", &min_duration_on,
               "if a segment is less than this value, then it is discarded. "
               "Set it to 0 so that no segment is discarded");

  po->Register("min-duration-off", &min_duration_off,
               "if the gap between to segments of the same speaker is less "
               "than this value, then these two segments are merged into a "
               "single segment. We do it recursively.");
}

// 'validate_models' checks the segmentation and embedding configs either
// against the file system or against a resource manager.
template <typename ValidateModels>
static bool ValidateOfflineSpeakerDiarizationConfig(
    const OfflineSpeakerDiarizationConfig &config,
    const ValidateModels &validate_models) {
  if (!validate_models()) {
    return false;
  }

  if (!config.clustering.Validate()) {
    return false;
  }

  if (config.min_duration_on < 0) {
    SHERPA_ONNX_LOGE("min_duration_on %.3f is negative",
                     config.min_duration_on);
    return false;
  }

  if (config.min_duration_off < 0) {
    SHERPA_ONNX_LOGE("min_duration_off %.3f is negative",
                     config.min_duration_off);
    return false;
  }

  return true;
}

bool OfflineSpeakerDiarizationConfig::Validate() const {
  return ValidateOfflineSpeakerDiarizationConfig(*this, [this]() -> bool {
    return segmentation.Validate() && embedding.Validate();
  });
}

template <typename Manager>
bool OfflineSpeakerDiarizationConfig::Validate(Manager *mgr) const {
  return ValidateOfflineSpeakerDiarizationConfig(*this, [this, mgr]() -> bool {
    return segmentation.Validate(mgr) && embedding.Validate(mgr);
  });
}

#if __ANDROID_API__ >= 9
template bool OfflineSpeakerDiarizationConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineSpeakerDiarizationConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineSpeakerDiarizationConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSpeakerDiarizationConfig(";
  os << "segmentation=" << segmentation.ToString() << ", ";
  os << "embedding=" << embedding.ToString() << ", ";
  os << "clustering=" << clustering.ToString() << ", ";
  os << "min_duration_on=" << min_duration_on << ", ";
  os << "min_duration_off=" << min_duration_off << ")";

  return os.str();
}

OfflineSpeakerDiarization::OfflineSpeakerDiarization(
    const OfflineSpeakerDiarizationConfig &config)
    : impl_(OfflineSpeakerDiarizationImpl::Create(config)) {}

template <typename Manager>
OfflineSpeakerDiarization::OfflineSpeakerDiarization(
    Manager *mgr, const OfflineSpeakerDiarizationConfig &config)
    : impl_(OfflineSpeakerDiarizationImpl::Create(mgr, config)) {}

OfflineSpeakerDiarization::~OfflineSpeakerDiarization() = default;

int32_t OfflineSpeakerDiarization::SampleRate() const {
  return impl_->SampleRate();
}

void OfflineSpeakerDiarization::SetConfig(
    const OfflineSpeakerDiarizationConfig &config) {
  impl_->SetConfig(config);
}

OfflineSpeakerDiarizationResult OfflineSpeakerDiarization::Process(
    const float *audio, int32_t n,
    OfflineSpeakerDiarizationProgressCallback callback /*= nullptr*/,
    void *callback_arg /*= nullptr*/) const {
  return impl_->Process(audio, n, std::move(callback), callback_arg);
}

std::unique_ptr<OfflineStream> OfflineSpeakerDiarization::CreateStream() const {
  SpeakerDiarizationTag tag;
  tag.sample_rate = SampleRate();
  return std::make_unique<OfflineStream>(tag);
}

OfflineSpeakerDiarizationResult OfflineSpeakerDiarization::Process(
    const OfflineStream *stream,
    OfflineSpeakerDiarizationProgressCallback callback /*= nullptr*/,
    void *callback_arg /*= nullptr*/) const {
  return impl_->Process(stream, std::move(callback), callback_arg);
}

#if __ANDROID_API__ >= 9
template OfflineSpeakerDiarization::OfflineSpeakerDiarization(
    AAssetManager *mgr, const OfflineSpeakerDiarizationConfig &config);
#endif

#if __OHOS__
template OfflineSpeakerDiarization::OfflineSpeakerDiarization(
    NativeResourceManager *mgr, const OfflineSpeakerDiarizationConfig &config);
#endif

}  // namespace sherpa_onnx
