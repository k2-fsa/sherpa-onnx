// sherpa-onnx/csrc/offline-dolphin-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-dolphin-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineDolphinModelConfig::Register(ParseOptions *po) {
  po->Register("dolphin-model", &model,
               "Path to model.onnx of Dolphin CTC branch.");
  po->Register("dolphin-encoder", &encoder,
               "Path to encoder.onnx of Dolphin attention decoder branch. "
               "Used only when --dolphin-decoder is provided.");
  po->Register("dolphin-decoder", &decoder,
               "Path to decoder.onnx of Dolphin attention decoder branch. "
               "When provided together with --dolphin-encoder, the attention "
               "decoder is used for recognition, which supports --dolphin-"
               "language and --dolphin-region.");
  po->Register("dolphin-language", &language,
               "Language code for the Dolphin attention decoder, e.g., zh, "
               "en, fil. Leave it empty to let the decoder predict it. Used "
               "only when --dolphin-decoder is provided.");
  po->Register("dolphin-region", &region,
               "Region code for the Dolphin attention decoder, e.g., CN, PH, "
               "US. It requires --dolphin-language to be set. Leave it empty "
               "to let the decoder predict it. Used only when "
               "--dolphin-decoder is provided.");
}

bool OfflineDolphinModelConfig::Validate() const {
  if (model.empty() && encoder.empty()) {
    SHERPA_ONNX_LOGE(
        "Please provide either --dolphin-model or --dolphin-encoder");
    return false;
  }

  if (!model.empty() && !FileExists(model)) {
    SHERPA_ONNX_LOGE("Dolphin model '%s' does not exist", model.c_str());
    return false;
  }

  if (!encoder.empty() && !FileExists(encoder)) {
    SHERPA_ONNX_LOGE("Dolphin encoder '%s' does not exist", encoder.c_str());
    return false;
  }

  if (!decoder.empty() && encoder.empty()) {
    SHERPA_ONNX_LOGE("--dolphin-decoder requires --dolphin-encoder");
    return false;
  }

  if (!encoder.empty() && decoder.empty()) {
    SHERPA_ONNX_LOGE("--dolphin-encoder requires --dolphin-decoder");
    return false;
  }

  if (!decoder.empty() && !FileExists(decoder)) {
    SHERPA_ONNX_LOGE("Dolphin decoder '%s' does not exist", decoder.c_str());
    return false;
  }

  if (decoder.empty() && (!language.empty() || !region.empty())) {
    SHERPA_ONNX_LOGE(
        "--dolphin-language and --dolphin-region are used only when "
        "--dolphin-decoder is provided");
    return false;
  }

  if (!region.empty() && language.empty()) {
    SHERPA_ONNX_LOGE("--dolphin-region requires --dolphin-language");
    return false;
  }

  return true;
}

std::string OfflineDolphinModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineDolphinModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "language=\"" << language << "\", ";
  os << "region=\"" << region << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
