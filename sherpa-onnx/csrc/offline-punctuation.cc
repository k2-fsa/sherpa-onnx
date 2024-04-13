// sherpa-onnx/csrc/offline-punctuation.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-punctuation.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-punctuation-impl.h"

namespace sherpa_onnx {

void OfflinePunctuationConfig::Register(ParseOptions *po) {
  model.Register(po);
}

bool OfflinePunctuationConfig::Validate() const {
  if (!model.Validate()) {
    return false;
  }

  return true;
}

std::string OfflinePunctuationConfig::ToString() const {
  std::ostringstream os;

  os << "OfflinePunctuationConfig(";
  os << "model=" << model.ToString() << ")";

  return os.str();
}

OfflinePunctuation::OfflinePunctuation(const OfflinePunctuationConfig &config)
    : impl_(OfflinePunctuationImpl::Create(config)) {}

OfflinePunctuation::~OfflinePunctuation() = default;

std::string OfflinePunctuation::AddPunctuation(const std::string &text) const {
  return impl_->AddPunctuation(text);
}

}  // namespace sherpa_onnx
