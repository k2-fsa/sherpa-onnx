// sherpa-onnx/csrc/offline-paraformer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-paraformer-model-config.h"

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineParaformerModelConfig::Register(ParseOptions *po) {
  po->Register(
      "paraformer", &model,
      "Path to model.onnx of Paraformer. If you use Ascend NPU, it is "
      "/path/to/encoder.om,/path/to/predictor.om,/path/to/decoder.om"
      "If you use RK NPU, it is "
      "/path/to/encoder.rknn,/path/to/predictor.rknn,/path/to/decoder.rknn");
}

bool OfflineParaformerModelConfig::Validate() const {
  if (EndsWith(model, ".onnx")) {
    if (!FileExists(model)) {
      SHERPA_ONNX_LOGE("Paraformer model '%s' does not exist", model.c_str());
      return false;
    }
    return true;
  }

  if (EndsWith(model, ".om")) {
    std::vector<std::string> filenames;
    SplitStringToVector(model, ",", false, &filenames);
    if (filenames.size() != 3 || !EndsWith(filenames[0], "encoder.om") ||
        !EndsWith(filenames[1], "predictor.om") ||
        !EndsWith(filenames[2], "decoder.om")) {
      SHERPA_ONNX_LOGE(
          "For Ascend NPU, you should pass "
          "/path/to/encoder.om,/path/to/predictor.om,/path/to/decoder.om. "
          "Given '%s'",
          model.c_str());
      return false;
    }

    return true;
  }

  if (EndsWith(model, ".rknn")) {
    std::vector<std::string> filenames;
    SplitStringToVector(model, ",", false, &filenames);
    if (filenames.size() != 3 || !EndsWith(filenames[0], "encoder.rknn") ||
        !EndsWith(filenames[1], "predictor.rknn") ||
        !EndsWith(filenames[2], "decoder.rknn")) {
      SHERPA_ONNX_LOGE(
          "For RKNN, you should pass "
          "/path/encoder.rknn,/path/predictor.rknn,/path/decoder.rknn. "
          "Given '%s'",
          model.c_str());
      return false;
    }

    return true;
  }

  SHERPA_ONNX_LOGE("Please pass *.onnx, *.om, or *.rknn models. Given '%s'",
                   model.c_str());
  return false;
}

std::string OfflineParaformerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineParaformerModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
