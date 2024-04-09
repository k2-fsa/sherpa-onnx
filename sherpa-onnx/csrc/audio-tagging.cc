// sherpa-onnx/csrc/audio-tagging.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging.h"

#include "sherpa-onnx/csrc/audio-tagging-impl.h"

namespace sherpa_onnx {

AudioTagging::AudioTagging(const AudioTaggingConfig &config)
    : impl_(AudioTaggingImpl::Create(config)) {}

AudioTagging::~AudioTagging() = default;

std::unique_ptr<OfflineStream> AudioTagging::CreateStream() const {
  return impl_->CreateStream();
}

std::vector<AudioEvent> AudioTagging::Compute(OfflineStream *s,
                                              int32_t top_k /*= -1*/) const {
  return impl_->Compute(s, top_k);
}

}  // namespace sherpa_onnx
