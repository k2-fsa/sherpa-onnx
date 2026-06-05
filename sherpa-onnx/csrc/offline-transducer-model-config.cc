// sherpa-onnx/csrc/offline-transducer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/offline-transducer-model-config.h"

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

bool IsQnnTransducerArtifact(const OfflineTransducerModelConfig &config) {
  return IsQnnModelLibFile(config.encoder_filename) ||
         IsQnnModelLibFile(config.decoder_filename) ||
         IsQnnModelLibFile(config.joiner_filename) ||
         !config.qnn_config.context_binary.empty();
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
        "For offline transducer with QNN, you should provide 3 context "
        "binaries separated by commas. Given '%s'",
        context_binary.c_str());
    return false;
  }

  return true;
}

void OfflineTransducerModelConfig::Register(ParseOptions *po) {
  po->Register("encoder", &encoder_filename, "Path to encoder.onnx");
  po->Register("decoder", &decoder_filename, "Path to decoder.onnx");
  po->Register("joiner", &joiner_filename, "Path to joiner.onnx");

  ParseOptions p("transducer", po);
  qnn_config.Register(&p);
}

bool OfflineTransducerModelConfig::Validate() const {
  bool uses_qnn = IsQnnTransducerArtifact(*this);

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
      if (!EndsWith(encoder_filename, ".so") ||
          !EndsWith(decoder_filename, ".so") ||
          !EndsWith(joiner_filename, ".so")) {
        SHERPA_ONNX_LOGE(
            "For offline transducer with QNN, encoder/decoder/joiner should be "
            "*.so when context binaries are missing. Given encoder: '%s', "
            "decoder: '%s', joiner: '%s'",
            encoder_filename.c_str(), decoder_filename.c_str(),
            joiner_filename.c_str());
        return false;
      }

      if (!FileExists(encoder_filename)) {
        SHERPA_ONNX_LOGE("transducer encoder: '%s' does not exist",
                         encoder_filename.c_str());
        return false;
      }

      if (!FileExists(decoder_filename)) {
        SHERPA_ONNX_LOGE("transducer decoder: '%s' does not exist",
                         decoder_filename.c_str());
        return false;
      }

      if (!FileExists(joiner_filename)) {
        SHERPA_ONNX_LOGE("transducer joiner: '%s' does not exist",
                         joiner_filename.c_str());
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

  if (!FileExists(encoder_filename)) {
    SHERPA_ONNX_LOGE("transducer encoder: '%s' does not exist",
                     encoder_filename.c_str());
    return false;
  }

  if (!FileExists(decoder_filename)) {
    SHERPA_ONNX_LOGE("transducer decoder: '%s' does not exist",
                     decoder_filename.c_str());
    return false;
  }

  if (!FileExists(joiner_filename)) {
    SHERPA_ONNX_LOGE("transducer joiner: '%s' does not exist",
                     joiner_filename.c_str());
    return false;
  }

  return true;
}

std::string OfflineTransducerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTransducerModelConfig(";
  os << "encoder_filename=\"" << encoder_filename << "\", ";
  os << "decoder_filename=\"" << decoder_filename << "\", ";
  os << "joiner_filename=\"" << joiner_filename << "\"";
  if (!qnn_config.backend_lib.empty()) {
    os << ", qnn_config=" << qnn_config.ToString();
  }
  os << ")";

  return os.str();
}

}  // namespace sherpa_onnx
