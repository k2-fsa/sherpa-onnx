// sherpa-onnx/csrc/offline-moonshine-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-moonshine-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

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
}

bool OfflineMoonshineModelConfig::Validate() const {
  // both v1 and v2 require a encoder model
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
          "--moonshine-merged_decoder for v2");
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
  os << "merged_decoder=\"" << merged_decoder << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
