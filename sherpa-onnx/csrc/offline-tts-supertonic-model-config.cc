// sherpa-onnx/csrc/offline-tts-supertonic-model-config.cc
//
// Copyright (c)  2026 zengyw

#include "sherpa-onnx/csrc/offline-tts-supertonic-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsSupertonicModelConfig::Register(ParseOptions *po) {
  po->Register("supertonic-duration-predictor", &duration_predictor,
               "Path to duration_predictor.onnx for Supertonic TTS");
  po->Register("supertonic-text-encoder", &text_encoder,
               "Path to text_encoder.onnx for Supertonic TTS");
  po->Register("supertonic-vector-estimator", &vector_estimator,
               "Path to vector_estimator.onnx for Supertonic TTS");
  po->Register("supertonic-vocoder", &vocoder,
               "Path to vocoder.onnx for Supertonic TTS");
  po->Register("supertonic-tts-json", &tts_json,
               "Path to tts.json for Supertonic TTS");
  po->Register("supertonic-unicode-indexer", &unicode_indexer,
               "Path to unicode_indexer.bin for Supertonic TTS");
  po->Register("supertonic-voice-style", &voice_style,
               "Path to Supertonic voice.bin (use sid 0..NumSpeakers()-1 to "
               "select)");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOfflineTtsSupertonicModelConfig(
    const OfflineTtsSupertonicModelConfig &config, const Exists &exists) {
  if (config.duration_predictor.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-duration-predictor");
    return false;
  }
  if (!exists(config.duration_predictor)) {
    SHERPA_ONNX_LOGE("--supertonic-duration-predictor '%s' does not exist",
                     config.duration_predictor.c_str());
    return false;
  }

  if (config.text_encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-text-encoder");
    return false;
  }
  if (!exists(config.text_encoder)) {
    SHERPA_ONNX_LOGE("--supertonic-text-encoder '%s' does not exist",
                     config.text_encoder.c_str());
    return false;
  }

  if (config.vector_estimator.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-vector-estimator");
    return false;
  }
  if (!exists(config.vector_estimator)) {
    SHERPA_ONNX_LOGE("--supertonic-vector-estimator '%s' does not exist",
                     config.vector_estimator.c_str());
    return false;
  }

  if (config.vocoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-vocoder");
    return false;
  }
  if (!exists(config.vocoder)) {
    SHERPA_ONNX_LOGE("--supertonic-vocoder '%s' does not exist",
                     config.vocoder.c_str());
    return false;
  }

  if (config.tts_json.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-tts-json");
    return false;
  }
  if (!exists(config.tts_json)) {
    SHERPA_ONNX_LOGE("--supertonic-tts-json '%s' does not exist",
                     config.tts_json.c_str());
    return false;
  }

  if (config.unicode_indexer.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-unicode-indexer");
    return false;
  }
  if (!exists(config.unicode_indexer)) {
    SHERPA_ONNX_LOGE("--supertonic-unicode-indexer '%s' does not exist",
                     config.unicode_indexer.c_str());
    return false;
  }

  if (config.voice_style.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-voice-style");
    return false;
  }
  if (!exists(config.voice_style)) {
    SHERPA_ONNX_LOGE("--supertonic-voice-style '%s' does not exist",
                     config.voice_style.c_str());
    return false;
  }
  return true;
}

bool OfflineTtsSupertonicModelConfig::Validate() const {
  return ValidateOfflineTtsSupertonicModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineTtsSupertonicModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineTtsSupertonicModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OfflineTtsSupertonicModelConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineTtsSupertonicModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OfflineTtsSupertonicModelConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineTtsSupertonicModelConfig(";
  os << "duration_predictor=\"" << duration_predictor << "\", ";
  os << "text_encoder=\"" << text_encoder << "\", ";
  os << "vector_estimator=\"" << vector_estimator << "\", ";
  os << "vocoder=\"" << vocoder << "\", ";
  os << "tts_json=\"" << tts_json << "\", ";
  os << "unicode_indexer=\"" << unicode_indexer << "\", ";
  os << "voice_style=\"" << voice_style << "\")";
  return os.str();
}

}  // namespace sherpa_onnx
