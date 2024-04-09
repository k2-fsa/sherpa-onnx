// sherpa-onnx/csrc/audio-tagging-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_IMPL_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_IMPL_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/audio-tagging.h"

namespace sherpa_onnx {

class AudioTaggingImpl {
 public:
  virtual ~AudioTaggingImpl() = default;

  static std::unique_ptr<AudioTaggingImpl> Create(
      const AudioTaggingConfig &config);

  virtual std::unique_ptr<OfflineStream> CreateStream() const = 0;

  virtual std::vector<AudioEvent> Compute(OfflineStream *s,
                                          int32_t top_k = -1) const = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_IMPL_H_
