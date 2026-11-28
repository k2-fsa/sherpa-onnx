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

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOfflineTtsZipvoiceModelConfig(
    const OfflineTtsZipvoiceModelConfig &config, const Exists &exists) {
  if (config.tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-tokens");
    return false;
  }
  if (!exists(config.tokens)) {
    SHERPA_ONNX_LOGE("--zipvoice-tokens: '%s' does not exist",
                     config.tokens.c_str());
    return false;
  }

  if (config.encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-encoder");
    return false;
  }
  if (!exists(config.encoder)) {
    SHERPA_ONNX_LOGE("--zipvoice-encoder: '%s' does not exist",
                     config.encoder.c_str());
    return false;
  }

  if (config.decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-decoder");
    return false;
  }
  if (!exists(config.decoder)) {
    SHERPA_ONNX_LOGE("--zipvoice-decoder: '%s' does not exist",
                     config.decoder.c_str());
    return false;
  }

  if (config.vocoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-vocoder");
    return false;
  }

  if (!exists(config.vocoder)) {
    SHERPA_ONNX_LOGE("--zipvoice-vocoder: '%s' does not exist",
                     config.vocoder.c_str());
    return false;
  }

  if (!config.data_dir.empty()) {
    SHERPA_ONNX_LOGE(
        "WARNING: --zipvoice-data-dir is deprecated and ignored in "
        "sherpa-onnx >= v2.0.0. Please use --zipvoice-lexicon or "
        "GenerationConfig.phoneme_codepoints with an external "
        "phonemizer (e.g. piper_phonemize) instead.");
  }

  if (!config.lexicon.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(config.lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!exists(f)) {
        SHERPA_ONNX_LOGE(
            "lexicon '%s' does not exist. Please check --zipvoice-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (config.feat_scale <= 0) {
    SHERPA_ONNX_LOGE("--zipvoice-feat-scale must be positive. Given: %f",
                     config.feat_scale);
    return false;
  }

  if (config.t_shift < 0) {
    SHERPA_ONNX_LOGE("--zipvoice-t-shift must be non-negative. Given: %f",
                     config.t_shift);
    return false;
  }

  if (config.target_rms <= 0) {
    SHERPA_ONNX_LOGE("--zipvoice-target-rms must be positive. Given: %f",
                     config.target_rms);
    return false;
  }

  if (config.guidance_scale <= 0) {
    SHERPA_ONNX_LOGE("--zipvoice-guidance-scale must be positive. Given: %f",
                     config.guidance_scale);
    return false;
  }

  return true;
}

bool OfflineTtsZipvoiceModelConfig::Validate() const {
  return ValidateOfflineTtsZipvoiceModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OfflineTtsZipvoiceModelConfig::Validate(Manager *mgr) const {
  return ValidateOfflineTtsZipvoiceModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OfflineTtsZipvoiceModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OfflineTtsZipvoiceModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

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
