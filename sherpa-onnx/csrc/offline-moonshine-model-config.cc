// sherpa-onnx/csrc/offline-moonshine-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-moonshine-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

static bool IsQnnModelLibFile(const std::string &filename) {
  return EndsWith(filename, ".so");
}

static bool IsQnnMoonshineArtifact(const OfflineMoonshineModelConfig &config) {
  return IsQnnModelLibFile(config.encoder) ||
         IsQnnModelLibFile(config.merged_decoder) ||
         !config.qnn_config.context_binary.empty();
}

static bool ValidateQnnContextBinaries(const std::string &context_binary,
                                       std::vector<std::string> &filenames) {
  filenames.clear();

  if (context_binary.empty()) {
    return true;
  }

  SplitStringToVector(context_binary, ",", true, &filenames);
  if (filenames.size() != 2) {
    SHERPA_ONNX_LOGE(
        "For moonshine with QNN, you should provide 2 context "
        "binaries separated by commas (encoder,decoder). Given '%s'",
        context_binary.c_str());
    return false;
  }

  return true;
}

void OfflineMoonshineModelConfig::Register(ParseOptions *po) {
  po->Register(
      "moonshine-preprocessor", &preprocessor,
      "Path to onnx preprocessor of moonshine v1, e.g., preprocess.onnx");

  po->Register("moonshine-encoder", &encoder,
               "Path to onnx encoder of moonshine v1 or v2, e.g., encode.onnx "
               "for v1, encoder_model.onnx for v2");

  po->Register("moonshine-uncached-decoder", &uncached_decoder,
               "Path to onnx uncached_decoder of moonshine v1, e.g., "
               "uncached_decode.onnx");

  po->Register(
      "moonshine-cached-decoder", &cached_decoder,
      "Path to onnx cached_decoder of moonshine v1, e.g., cached_decode.onnx");

  po->Register("moonshine-merged-decoder", &merged_decoder,
               "Path to onnx merged decoder of moonshine v2, e.g., "
               "decoder_model_merged.onnx");

  std::string prefix = "moonshine";
  ParseOptions p(prefix, po);
  qnn_config.Register(&p);
}

bool OfflineMoonshineModelConfig::Validate() const {
  bool uses_qnn = IsQnnMoonshineArtifact(*this);

  if (uses_qnn) {
    std::vector<std::string> context_binaries;
    if (!ValidateQnnContextBinaries(qnn_config.context_binary,
                                    context_binaries)) {
      return false;
    }

    if (!qnn_config.Validate()) {
      return false;
    }

    // When using context binaries, encoder/decoder lib files are not required
    if (qnn_config.context_binary.empty()) {
      // Need model lib files if no context binary
      if (encoder.empty()) {
        SHERPA_ONNX_LOGE("Please provide --moonshine-encoder");
        return false;
      }

      if (merged_decoder.empty()) {
        SHERPA_ONNX_LOGE(
            "Please provide --moonshine-merged-decoder for QNN");
        return false;
      }
    }

    return true;
  }

  // Non-QNN path: both v1 and v2 require an encoder model
  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --moonshine-encoder");
    return false;
  }

  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("moonshine encoder file '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (merged_decoder.empty()) {
    // for v1
    if (preprocessor.empty()) {
      SHERPA_ONNX_LOGE(
          "Please provide --moonshine-preprocessor for v1 or "
          "--moonshine-merged-decoder for v2");
      return false;
    }

    if (!FileExists(preprocessor)) {
      SHERPA_ONNX_LOGE("moonshine preprocessor file '%s' does not exist",
                       preprocessor.c_str());
      return false;
    }

    if (uncached_decoder.empty()) {
      SHERPA_ONNX_LOGE("Please provide --moonshine-uncached-decoder for v1");
      return false;
    }

    if (!FileExists(uncached_decoder)) {
      SHERPA_ONNX_LOGE("moonshine uncached decoder file '%s' does not exist",
                       uncached_decoder.c_str());
      return false;
    }

    if (cached_decoder.empty()) {
      SHERPA_ONNX_LOGE("Please provide --moonshine-cached-decoder for v1");
      return false;
    }

    if (!FileExists(cached_decoder)) {
      SHERPA_ONNX_LOGE("moonshine cached decoder file '%s' does not exist",
                       cached_decoder.c_str());
      return false;
    }
  } else {
    // v2
    if (!preprocessor.empty()) {
      SHERPA_ONNX_LOGE("Please don't provide preprocessor for moonshine v2");
      return false;
    }

    if (!uncached_decoder.empty()) {
      SHERPA_ONNX_LOGE(
          "Please don't provide uncached decoder for moonshine v2");
      return false;
    }

    if (!cached_decoder.empty()) {
      SHERPA_ONNX_LOGE("Please don't provide cached decoder for moonshine v2");
      return false;
    }

    if (!FileExists(merged_decoder)) {
      SHERPA_ONNX_LOGE(
          "moonshine v2 merged_decoder decoder file '%s' does not exist",
          merged_decoder.c_str());
      return false;
    }
  }

  return true;
}

std::string OfflineMoonshineModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineMoonshineModelConfig(";
  os << "preprocessor=\"" << preprocessor << "\", ";
  os << "encoder=\"" << encoder << "\", ";
  os << "uncached_decoder=\"" << uncached_decoder << "\", ";
  os << "cached_decoder=\"" << cached_decoder << "\", ";
  os << "merged_decoder=\"" << merged_decoder << "\", ";
  if (!qnn_config.backend_lib.empty()) {
    os << "qnn_config=" << qnn_config.ToString() << ", ";
  }
  os << ")";

  return os.str();
}

}  // namespace sherpa_onnx
