// sherpa-onnx/csrc/offline-tts-zipvoice-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-zipvoice-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineTtsZipvoiceModelConfig::Register(ParseOptions *po) {
  po->Register("zipvoice-tokens", &tokens,
               "Path to tokens.txt for ZipVoice models");
  po->Register("zipvoice-data-dir", &data_dir,
               "Deprecated. Ignored since v2.0.0: espeak-ng is no longer used. "
               "Use --zipvoice-lexicon or "
               "GenerationConfig.phoneme_codepoints instead.");
  po->Register("zipvoice-lexicon", &lexicon, "Path to lexicon.txt");
  po->Register("zipvoice-encoder", &encoder, "Path to zipvoice text model");
  po->Register("zipvoice-decoder", &decoder,
               "Path to zipvoice flow-matching decoder model");
  po->Register("zipvoice-vocoder", &vocoder, "Path to zipvoice vocoder");
  po->Register("zipvoice-feat-scale", &feat_scale,
               "Feature scale for ZipVoice (default: 0.1)");
  po->Register("zipvoice-t-shift", &t_shift,
               "Shift t to smaller ones if t_shift < 1.0 (default: 0.5)");
  po->Register(
      "zipvoice-target-rms", &target_rms,
      "Target speech normalization rms value for ZipVoice (default: 0.1)");
  po->Register(
      "zipvoice-guidance-scale", &guidance_scale,
      "The scale of classifier-free guidance during inference for ZipVoice "
      "(default: 1.0)");
}

bool OfflineTtsZipvoiceModelConfig::Validate() const {
  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-tokens");
    return false;
  }
  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--zipvoice-tokens: '%s' does not exist", tokens.c_str());
    return false;
  }

  if (encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-encoder");
    return false;
  }
  if (!FileExists(encoder)) {
    SHERPA_ONNX_LOGE("--zipvoice-encoder: '%s' does not exist",
                     encoder.c_str());
    return false;
  }

  if (decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-decoder");
    return false;
  }
  if (!FileExists(decoder)) {
    SHERPA_ONNX_LOGE("--zipvoice-decoder: '%s' does not exist",
                     decoder.c_str());
    return false;
  }

  if (vocoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-vocoder");
    return false;
  }

  if (!FileExists(vocoder)) {
    SHERPA_ONNX_LOGE("--zipvoice-vocoder: '%s' does not exist",
                     vocoder.c_str());
    return false;
  }

  if (!data_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "WARNING: --zipvoice-data-dir is deprecated and ignored in "
        "sherpa-onnx >= v2.0.0. Please use --zipvoice-lexicon or "
        "GenerationConfig.phoneme_codepoints with an external "
        "phonemizer (e.g. piper_phonemize) instead.");
  }

  if (!lexicon.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE(
            "lexicon '%s' does not exist. Please check --zipvoice-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (feat_scale <= 0) {
    SHERPA_ONNX_LOGE("--zipvoice-feat-scale must be positive. Given: %f",
                     feat_scale);
    return false;
  }

  if (t_shift < 0) {
    SHERPA_ONNX_LOGE("--zipvoice-t-shift must be non-negative. Given: %f",
                     t_shift);
    return false;
  }

  if (target_rms <= 0) {
    SHERPA_ONNX_LOGE("--zipvoice-target-rms must be positive. Given: %f",
                     target_rms);
    return false;
  }

  if (guidance_scale <= 0) {
    SHERPA_ONNX_LOGE("--zipvoice-guidance-scale must be positive. Given: %f",
                     guidance_scale);
    return false;
  }

  return true;
}

std::string OfflineTtsZipvoiceModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsZipvoiceModelConfig(";
  os << "tokens=\"" << tokens << "\", ";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\", ";
  os << "vocoder=\"" << vocoder << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "feat_scale=" << feat_scale << ", ";
  os << "t_shift=" << t_shift << ", ";
  os << "target_rms=" << target_rms << ", ";
  os << "guidance_scale=" << guidance_scale << ")";

  return os.str();
}

}  // namespace sherpa_onnx
