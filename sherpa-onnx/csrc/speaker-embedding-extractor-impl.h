// sherpa-onnx/csrc/speaker-embedding-extractor-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorImpl {
 public:
  virtual ~SpeakerEmbeddingExtractorImpl() = default;

  static std::unique_ptr<SpeakerEmbeddingExtractorImpl> Create(
      const SpeakerEmbeddingExtractorConfig &config);

#if __ANDROID_API__ >= 9
  static std::unique_ptr<SpeakerEmbeddingExtractorImpl> Create(
      AAssetManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

  virtual int32_t Dim() const = 0;

  virtual std::unique_ptr<OnlineStream> CreateStream() const = 0;

  virtual bool IsReady(OnlineStream *s) const = 0;

  virtual std::vector<float> Compute(OnlineStream *s) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_IMPL_H_
