// sherpa-onnx/csrc/online-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-transducer-model-config.h"

#include <sstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

static bool IsQnnModelLibFile(const std::string &filename) {
  return EndsWith(filename, ".so");
}

static bool ValidateQnnContextBinaries(const std::string &context_binary,
                                       std::vector<std::string> &filenames) {
  filenames.clear();

  if (context_binary.empty()) {
    return true;
  }

  SplitStringToVector(context_binary, ",", true, &filenames);
  if (filenames.size() != 3) {
    SHERPA_ONNX_LOGE(
        "For online transducer with QNN, you should provide 3 context "
        "binaries separated by commas. Given '%s'",
        context_binary.c_str());
    return false;
  }

  return true;
}

void OnlineTransducerModelConfig::Register(ParseOptions *po) {
  po->Register("encoder", &encoder, "Path to encoder.onnx");
  po->Register("decoder", &decoder, "Path to decoder.onnx");
  po->Register("joiner", &joiner, "Path to joiner.onnx");

  ParseOptions p("transducer", po);
  qnn_config.Register(&p);
}

bool OnlineTransducerModelConfig::Validate() const {
  bool uses_qnn = IsQnnModelLibFile(encoder) || IsQnnModelLibFile(decoder) ||
                  IsQnnModelLibFile(joiner) ||
                  !qnn_config.context_binary.empty();

  if (uses_qnn) {
    std::vector<std::string> context_binaries;
    if (!ValidateQnnContextBinaries(qnn_config.context_binary,
                                    context_binaries)) {
      return false;
    }

    bool need_model_libs = context_binaries.empty();
    for (const auto &name : context_binaries) {
      if (!FileExists(name)) {
        need_model_libs = true;
        break;
      }
    }

    if (need_model_libs) {
      if (!EndsWith(encoder, ".so") || !EndsWith(decoder, ".so") ||
          !EndsWith(joiner, ".so")) {
        SHERPA_ONNX_LOGE(
            "For online transducer with QNN, encoder/decoder/joiner should be "
            "*.so when context binaries are missing. Given encoder: '%s', "
            "decoder: '%s', joiner: '%s'",
            encoder.c_str(), decoder.c_str(), joiner.c_str());
        return false;
      }

      if (!FileExists(encoder)) {
        SHERPA_ONNX_LOGE("transducer encoder: '%s' does not exist",
                         encoder.c_str());
        return false;
      }

      if (!FileExists(decoder)) {
        SHERPA_ONNX_LOGE("transducer decoder: '%s' does not exist",
                         decoder.c_str());
        return false;
      }

      if (!FileExists(joiner)) {
        SHERPA_ONNX_LOGE("joiner: '%s' does not exist", joiner.c_str());
        return false;
      }
    }

    for (const auto &name : context_binaries) {
      if (FileExists(name) && !EndsWith(name, ".bin")) {
        SHERPA_ONNX_LOGE("QNN context binary should end with .bin. Given '%s'",
                         name.c_str());
        return false;
      }
    }

    return qnn_config.Validate();
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("transducer encoder: '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("transducer decoder: '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  if (!FileExists(joiner)) {
    SHERPA_ONNX_LOGE("joiner: '%s' does not exist", joiner.c_str());
    return false;
  }

  return true;
}

std::string OnlineTransducerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineTransducerModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "joiner=\"" << joiner << "\"";
  if (!qnn_config.backend_lib.empty()) {
    os << ", qnn_config=" << qnn_config.ToString();
  }
  os << ")";

  return os.str();
}

}  // namespace sherpa_onnx
