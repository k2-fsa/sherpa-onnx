// sherpa-onnx/csrc/offline-tts-synthesizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-synthesizer.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-synthesizer-impl.h"

namespace sherpa_onnx {

void OfflineTtsSynthesizerConfig::Register(ParseOptions *po) {
  vits.Register(po);
}

bool OfflineTtsSynthesizerConfig::Validate() const { return vits.Validate(); }

std::string OfflineTtsSynthesizerConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsSynthesizerConfig(";
  os << "vits=" << vits.ToString() << ")";

  return os.str();
}

OfflineTtsSynthesizer::OfflineTtsSynthesizer(
    const OfflineTtsSynthesizerConfig &config)
    : impl_(OfflineTtsSynthesizerImpl::Create(config)) {}

OfflineTtsSynthesizer::~OfflineTtsSynthesizer() = default;

GeneratedAudio OfflineTtsSynthesizer::Generate(const std::string &text) const {
  return impl_->Generate(text);
}

}  // namespace sherpa_onnx
