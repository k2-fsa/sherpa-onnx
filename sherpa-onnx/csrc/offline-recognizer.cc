// sherpa-onnx/csrc/offline-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-recognizer.h"

#include <memory>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-lm-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"

namespace sherpa_onnx {

void OfflineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);
  lm_config.Register(po);

  po->Register(
      "decoding-method", &decoding_method,
      "decoding method,"
      "Valid values: greedy_search, modified_beam_search. "
      "modified_beam_search is applicable only for transducer models.");

  po->Register("max-active-paths", &max_active_paths,
               "Used only when decoding_method is modified_beam_search");
}

bool OfflineRecognizerConfig::Validate() const {
  if (decoding_method == "modified_beam_search" && !lm_config.model.empty()) {
    if (max_active_paths <= 0) {
      SHERPA_ONNX_LOGE("max_active_paths is less than 0! Given: %d",
                       max_active_paths);
      return false;
    }
    if (!lm_config.Validate()) return false;
  }

  return model_config.Validate();
}

std::string OfflineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "lm_config=" << lm_config.ToString() << ", ";
  os << "decoding_method=\"" << decoding_method << "\", ";
  os << "max_active_paths=" << max_active_paths << ")";

  return os.str();
}

OfflineRecognizer::OfflineRecognizer(const OfflineRecognizerConfig &config)
    : impl_(OfflineRecognizerImpl::Create(config)) {}

OfflineRecognizer::~OfflineRecognizer() = default;

std::unique_ptr<OfflineStream> OfflineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

void OfflineRecognizer::DecodeStreams(OfflineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

}  // namespace sherpa_onnx
