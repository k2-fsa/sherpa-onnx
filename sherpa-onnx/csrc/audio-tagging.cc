// sherpa-onnx/csrc/audio-tagging.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging.h"

#include "sherpa-onnx/csrc/audio-tagging-impl.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

std::string AudioEvent::ToString() const {
  std::ostringstream os;
  os << "AudioEvent(";
  os << "name=\"" << name << "\", ";
  os << "index=" << index << ", ";
  os << "prob=" << prob << ")";
  return os.str();
}

void AudioTaggingConfig::Register(ParseOptions *po) {
  model.Register(po);
  po->Register("top-k", &top_k, "Top k events to return in the result");
}

bool AudioTaggingConfig::Validate() const {
  if (!model.Validate()) {
    return false;
  }

  if (top_k < 1) {
    SHERPA_ONNX_LOGE("--top-k should be >= 1. Given: %d", top_k);
    return false;
  }

  return true;
}
std::string AudioTaggingConfig::ToString() const {
  std::ostringstream os;

  os << "AudioTaggingConfig(";
  os << "model=" << model.ToString() << ", ";
  os << "top_k=" << top_k << ")";

  return os.str();
}

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
