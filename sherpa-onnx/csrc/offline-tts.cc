// sherpa-onnx/csrc/offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts.h"

#include <string>

#include "sherpa-onnx/csrc/offline-tts-impl.h"

namespace sherpa_onnx {

void OfflineTtsConfig::Register(ParseOptions *po) { vits.Register(po); }

bool OfflineTtsConfig::Validate() const { return vits.Validate(); }

std::string OfflineTtsConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsConfig(";
  os << "vits=" << vits.ToString() << ")";

  return os.str();
}

OfflineTts::OfflineTts(const OfflineTtsConfig &config)
    : impl_(OfflineTtsImpl::Create(config)) {}

OfflineTts::~OfflineTts() = default;

GeneratedAudio OfflineTts::Generate(const std::string &text) const {
  return impl_->Generate(text);
}

}  // namespace sherpa_onnx
