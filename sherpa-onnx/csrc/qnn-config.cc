// sherpa-onnx/csrc/qnn-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void QnnConfig::Register(ParseOptions *po) {
  po->Register("qnn-backend-lib", &backend_lib,
               "Path to libQnnHtp.so "
               "Used only when provider is qnn."
               "Leave it empty if you don't use qnn");

  po->Register(
      "qnn-context-binary", &context_binary,
      "Path to model.bin. Used only when provider is qnn."
      "If it exists, libmodel.so is ignored."
      "If it does not exist, Context binary is saved to this path so that "
      "it is loaded the next time you run it. You can leave it empty if you "
      "don't use qnn");

  po->Register("qnn-system-lib", &system_lib,
               "Required and used only when --qnn-context-binary is not empty "
               "and exists. You can leave it empty if you don't use qnn.");
}

bool QnnConfig::Validate() const {
  if (backend_lib.empty()) {
    SHERPA_ONNX_LOGE("Please provide path to libQnnHtp.so if you use qnn");
    return false;
  }

  if (!FileExists(backend_lib)) {
    SHERPA_ONNX_LOGE("--qnn-backend-lib: '%s' does not exist",
                     backend_lib.c_str());
    return false;
  }

  if (!context_binary.empty() && FileExists(context_binary)) {
    if (system_lib.empty()) {
      SHERPA_ONNX_LOGE(
          "Please provide --qnn-system-lib when you provide "
          "--qnn-context-binary");
      return false;
    }

    if (!FileExists(system_lib)) {
      SHERPA_ONNX_LOGE("--qnn-system-lib: '%s' does not exist",
                       system_lib.c_str());
      return false;
    }
  }

  return true;
}

std::string QnnConfig::ToString() const {
  std::ostringstream os;

  os << "QnnConfig(";
  os << "backend_lib=\"" << backend_lib << "\", ";
  os << "context_binary=\"" << context_binary << "\", ";
  os << "system_lib=\"" << system_lib << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
