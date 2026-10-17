// sherpa-onnx/csrc/tts-text-normalizer.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_TTS_TEXT_NORMALIZER_H_
#define SHERPA_ONNX_CSRC_TTS_TEXT_NORMALIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "kaldifst/csrc/text-normalizer.h"

namespace sherpa_onnx {

std::vector<std::unique_ptr<kaldifst::TextNormalizer>> LoadTextNormalizers(
    const std::string &rule_fsts, const std::string &rule_fars, bool debug);

template <typename Manager>
std::vector<std::unique_ptr<kaldifst::TextNormalizer>> LoadTextNormalizers(
    Manager *mgr, const std::string &rule_fsts, const std::string &rule_fars,
    bool debug);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TTS_TEXT_NORMALIZER_H_
