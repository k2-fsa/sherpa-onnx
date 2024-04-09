// sherpa-onnx/csrc/audio-tagging-zipformer-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_ZIPFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_ZIPFORMER_IMPL_H_

#include <memory>

#include "sherpa-onnx/csrc/audio-tagging-impl.h"
#include "sherpa-onnx/csrc/audio-tagging.h"

namespace sherpa_onnx {

class AudioTaggingZipformerImpl : public AudioTaggingImpl {
 public:
  explicit AudioTaggingZipformerImpl(const AudioTaggingConfig &config)
      : config_(config) {}

  std::unique_ptr<OfflineStream> CreateStream() const override { return {}; }

  std::vector<AudioEvent> Compute(OfflineStream *s,
                                  int32_t top_k = -1) const override {
    return {};
  }

 private:
  AudioTaggingConfig config_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_ZIPFORMER_IMPL_H_
