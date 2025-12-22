// sherpa-onnx/csrc/offline-zipformer-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-zipformer-ctc-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineZipformerCtcModelConfig::Register(ParseOptions *po) {
  po->Register("zipformer-ctc-model", &model, "Path to zipformer CTC model");

  std::string prefix = "zipformer-ctc";
  ParseOptions p(prefix, po);

  qnn_config.Register(&p);
}

bool OfflineZipformerCtcModelConfig::Validate() const {
  if (qnn_config.context_binary.empty()) {
    if (model.empty()) {
      SHERPA_ONNX_LOGE("Please provide a Zipformer CTC model");
      return false;
    }

    if (!FileExists(model)) {
      SHERPA_ONNX_LOGE("Zipformer CTC model '%s' does not exist",
                       model.c_str());
      return false;
    }
  }

  if (model.empty() && !qnn_config.context_binary.empty()) {
    // we require that the context_binary exists
    if (!FileExists(qnn_config.context_binary)) {
      SHERPA_ONNX_LOGE(
          "Model is empty, but you provide a context binary that does not "
          "exist");
      return false;
    }
  }

  if (EndsWith(model, ".so") || EndsWith(model, ".bin") ||
      (model.empty() && !qnn_config.context_binary.empty())) {
    return qnn_config.Validate();
  }

  return true;
}

std::string OfflineZipformerCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineZipformerCtcModelConfig(";
  os << "model=\"" << model << "\"";

  if (!qnn_config.backend_lib.empty()) {
    os << ", qnn_config=" << qnn_config.ToString() << ", ";
  }

  os << ")";

  return os.str();
}

}  // namespace sherpa_onnx
