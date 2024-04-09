// sherpa-onnx/csrc/audio-tagging-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/audio-tagging-model-config.h"

namespace sherpa_onnx {

void AudioTaggingModelConfig::Register(ParseOptions *po) {
  zipformer.Register(po);
}

bool AudioTaggingModelConfig::Validate() const {
  if (!zipformer.model.empty() && !zipformer.Validate()) {
    return false;
  }

  return true;
}

std::string AudioTaggingModelConfig::ToString() const {
  std::ostringstream os;

  os << "AudioTaggingModelConfig(";
  os << "zipformer=" << zipformer.ToString() << ")";

  return os.str();
}

}  // namespace sherpa_onnx
