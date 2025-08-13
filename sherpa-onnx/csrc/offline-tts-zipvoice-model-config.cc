// sherpa-onnx/csrc/offline-tts-zipvoice-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-zipvoice-model-config.h"

#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsZipvoiceModelConfig::Register(ParseOptions *po) {
  po->Register("zipvoice-tokens", &tokens,
               "Path to tokens.txt for ZipVoice models");
  po->Register("zipvoice-data-dir", &data_dir,
               "Path to the directory containing dict for espeak-ng.");
  po->Register("zipvoice-pinyin-dict", &pinyin_dict,
               "Path to the pinyin dictionary for cppinyin (i.e converting "
               "Chinese into phones).");
  po->Register("zipvoice-text-model", &text_model,
               "Path to zipvoice text model");
  po->Register("zipvoice-flow-matching-model", &flow_matching_model,
               "Path to zipvoice flow-matching model");
  po->Register("zipvoice-vocoder", &vocoder, "Path to zipvoice vocoder");
  po->Register("zipvoice-num-step", &num_step,
               "Number of inference steps for ZipVoice (default: 16)");
  po->Register("zipvoice-feat-scale", &feat_scale,
               "Feature scale for ZipVoice (default: 0.1)");
  po->Register("zipvoice-speed", &speed,
               "Speech speed for ZipVoice (default: 1.0, larger=faster, "
               "smaller=slower)");
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

  if (text_model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-text-model");
    return false;
  }
  if (!FileExists(text_model)) {
    SHERPA_ONNX_LOGE("--zipvoice-text-model: '%s' does not exist",
                     text_model.c_str());
    return false;
  }

  if (flow_matching_model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --zipvoice-flow-matching-model");
    return false;
  }
  if (!FileExists(flow_matching_model)) {
    SHERPA_ONNX_LOGE("--zipvoice-flow-matching-model: '%s' does not exist",
                     flow_matching_model.c_str());
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
    std::vector<std::string> required_files = {
        "phontab",
        "phonindex",
        "phondata",
        "intonations",
    };
    for (const auto &f : required_files) {
      if (!FileExists(data_dir + "/" + f)) {
        SHERPA_ONNX_LOGE(
            "'%s/%s' does not exist. Please check zipvoice-data-dir",
            data_dir.c_str(), f.c_str());
        return false;
      }
    }
  }

  if (!pinyin_dict.empty() && !FileExists(pinyin_dict)) {
    SHERPA_ONNX_LOGE("--zipvoice-pinyin-dict: '%s' does not exist",
                     pinyin_dict.c_str());
    return false;
  }

  if (num_step <= 0) {
    SHERPA_ONNX_LOGE("--zipvoice-num-step must be positive. Given: %d",
                     num_step);
    return false;
  }

  if (feat_scale <= 0) {
    SHERPA_ONNX_LOGE("--zipvoice-feat-scale must be positive. Given: %f",
                     feat_scale);
    return false;
  }

  if (speed <= 0) {
    SHERPA_ONNX_LOGE("--zipvoice-speed must be positive. Given: %f", speed);
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
  os << "text_model=\"" << text_model << "\", ";
  os << "flow_matching_model=\"" << flow_matching_model << "\", ";
  os << "vocoder=\"" << vocoder << "\", ";
  os << "data_dir=\"" << data_dir << "\", ";
  os << "pinyin_dict=\"" << pinyin_dict << "\", ";
  os << "num_step=" << num_step << ", ";
  os << "feat_scale=" << feat_scale << ", ";
  os << "speed=" << speed << ", ";
  os << "t_shift=" << t_shift << ", ";
  os << "target_rms=" << target_rms << ", ";
  os << "guidance_scale=" << guidance_scale << ")";

  return os.str();
}

}  // namespace sherpa_onnx
