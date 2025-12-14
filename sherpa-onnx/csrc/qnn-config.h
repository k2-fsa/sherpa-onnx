// sherpa-onnx/csrc/qnn-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_QNN_CONFIG_H_
#define SHERPA_ONNX_CSRC_QNN_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct QnnConfig {
  // Path to the backend library, e.g.,
  // /some/path/to/libQnnHtp.so
  std::string backend_lib;

  // If it exists, you need to also provide system_lib.
  // In this case, the model lib, i.e., libmodel.so, is ignored
  //
  // If it does not exist and if the user want to save the context binary,
  // it will save it to this path.
  std::string context_binary;

  // Required and used only when context_binary exists
  // Example value: /some/path/to/libQnnSystem.so
  std::string system_lib;

  std::string ToString() const;

  void Register(ParseOptions *po);

  bool Validate() const;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_QNN_CONFIG_H_
