// sherpa-onnx/csrc/offline-tts-supertonic-model-config.cc
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/csrc/offline-tts-supertonic-model-config.h"

#include <sstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineTtsSupertonicModelConfig::Register(ParseOptions *po) {
  po->Register("supertonic-duration-predictor", &duration_predictor,
               "Path to duration_predictor.onnx");
  po->Register("supertonic-text-encoder", &text_encoder,
               "Path to text_encoder.onnx");
  po->Register("supertonic-vector-estimator", &vector_estimator,
               "Path to vector_estimator.onnx");
  po->Register("supertonic-vocoder", &vocoder, "Path to vocoder.onnx");
  po->Register(
      "supertonic-tts-config", &tts_config,
      "Path to TTS config file (use generate_tts_bin.py from tts.json)");
  po->Register("supertonic-unicode-indexer", &unicode_indexer,
               "Path to unicode_indexer.bin (use generate_indexer_bin.py from "
               "unicode_indexer.json)");
  po->Register("supertonic-voice-style", &voice_style,
               "Path to Supertonic voice.bin (multi-speaker: single file with "
               "multiple speakers; use sid 0..NumSpeakers()-1 to select)");
}

bool OfflineTtsSupertonicModelConfig::Validate() const {
  if (duration_predictor.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-duration-predictor");
    return false;
  }
  std::string abs_dp = ResolveAbsolutePath(duration_predictor);
  if (!FileExists(abs_dp)) {
    SHERPA_ONNX_LOGE("duration_predictor file does not exist: '%s'",
                     duration_predictor.c_str());
    return false;
  }

  if (text_encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-text-encoder");
    return false;
  }
  std::string abs_te = ResolveAbsolutePath(text_encoder);
  if (!FileExists(abs_te)) {
    SHERPA_ONNX_LOGE("text_encoder file does not exist: '%s'",
                     text_encoder.c_str());
    return false;
  }

  if (vector_estimator.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-vector-estimator");
    return false;
  }
  std::string abs_ve = ResolveAbsolutePath(vector_estimator);
  if (!FileExists(abs_ve)) {
    SHERPA_ONNX_LOGE("vector_estimator file does not exist: '%s'",
                     vector_estimator.c_str());
    return false;
  }

  if (vocoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-vocoder");
    return false;
  }
  std::string abs_voc = ResolveAbsolutePath(vocoder);
  if (!FileExists(abs_voc)) {
    SHERPA_ONNX_LOGE("vocoder file does not exist: '%s'", vocoder.c_str());
    return false;
  }

  if (tts_config.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-tts-config");
    return false;
  }
  std::string abs_tts_config = ResolveAbsolutePath(tts_config);
  if (!FileExists(abs_tts_config)) {
    SHERPA_ONNX_LOGE("TTS config file does not exist: '%s'",
                     tts_config.c_str());
    return false;
  }
  if (unicode_indexer.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-unicode-indexer");
    return false;
  }
  std::string abs_indexer = ResolveAbsolutePath(unicode_indexer);
  if (!FileExists(abs_indexer)) {
    SHERPA_ONNX_LOGE("unicode_indexer file does not exist: '%s'",
                     unicode_indexer.c_str());
    return false;
  }
  if (voice_style.empty()) {
    SHERPA_ONNX_LOGE("Please provide --supertonic-voice-style");
    return false;
  }
  std::vector<std::string> files;
  SplitStringToVector(voice_style, ",", false, &files);
  for (const auto &f : files) {
    std::string abs_path = ResolveAbsolutePath(f);
    if (!FileExists(abs_path)) {
      SHERPA_ONNX_LOGE("Voice style file does not exist: '%s'",
                       abs_path.c_str());
      return false;
    }
  }
  return true;
}

std::string OfflineTtsSupertonicModelConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineTtsSupertonicModelConfig(";
  os << "duration_predictor=\"" << duration_predictor << "\", ";
  os << "text_encoder=\"" << text_encoder << "\", ";
  os << "vector_estimator=\"" << vector_estimator << "\", ";
  os << "vocoder=\"" << vocoder << "\", ";
  os << "tts_config=\"" << tts_config << "\", ";
  os << "unicode_indexer=\"" << unicode_indexer << "\", ";
  os << "voice_style=\"" << voice_style << "\")";
  return os.str();
}

}  // namespace sherpa_onnx
