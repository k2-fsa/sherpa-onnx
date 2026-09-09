// sherpa-onnx/csrc/offline-tts-pocket-zh-en-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-pocket-zh-en-model-config.h"

#include <sstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineTtsPocketZhEnModelConfig::Register(ParseOptions *po) {
  po->Register("pocket-zh-en-step-model", &step_model,
               "Path to step_model.onnx for PocketTTS ZhEn");
  po->Register("pocket-zh-en-step-encoder", &step_encoder,
               "Path to step_encoder.onnx for PocketTTS ZhEn");
  po->Register("pocket-zh-en-lexicon", &lexicon,
               "Path to lexicon file(s) for PocketTTS ZhEn. "
               "You can pass multiple files separated by comma, e.g., "
               "lexicon-zh.txt,lexicon-en.txt");
  po->Register("pocket-zh-en-voice-embedding-cache-capacity",
               &voice_embedding_cache_capacity,
               "Capacity of the voice embedding cache (number of items). "
               "Default: 50. 0 disables caching.");
}

bool OfflineTtsPocketZhEnModelConfig::Validate() const {
  if (step_model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-zh-en-step-model");
    return false;
  }

  if (!FileExists(step_model)) {
    SHERPA_ONNX_LOGE("--pocket-zh-en-step-model '%s' does not exist",
                     step_model.c_str());
    return false;
  }

  if (step_encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-zh-en-step-encoder");
    return false;
  }

  if (!FileExists(step_encoder)) {
    SHERPA_ONNX_LOGE("--pocket-zh-en-step-encoder '%s' does not exist",
                     step_encoder.c_str());
    return false;
  }

  if (lexicon.empty()) {
    SHERPA_ONNX_LOGE("Please provide --pocket-zh-en-lexicon");
    return false;
  }

  {
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE(
            "--pocket-zh-en-lexicon '%s' does not exist. "
            "Please re-check --pocket-zh-en-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (voice_embedding_cache_capacity < 0) {
    SHERPA_ONNX_LOGE(
        "voice_embedding_cache_capacity must be non-negative. Given: %d",
        voice_embedding_cache_capacity);
    return false;
  }

  return true;
}

std::string OfflineTtsPocketZhEnModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsPocketZhEnModelConfig(";
  os << "step_model=\"" << step_model << "\", ";
  os << "step_encoder=\"" << step_encoder << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "voice_embedding_cache_capacity=" << voice_embedding_cache_capacity
     << ")";

  return os.str();
}

}  // namespace sherpa_onnx
