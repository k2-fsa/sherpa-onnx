// sherpa-onnx/csrc/tts-text-normalizer.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Shared helpers for loading text normalizer FSTs/FARs, used by multiple TTS
// implementations (Pocket TTS, Pocket ZhEn TTS, Kokoro TTS, etc.).

#ifndef SHERPA_ONNX_CSRC_TTS_TEXT_NORMALIZER_H_
#define SHERPA_ONNX_CSRC_TTS_TEXT_NORMALIZER_H_

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/fst-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

// Load text normalizers from rule_fsts and rule_fars (file path version).
inline std::vector<std::unique_ptr<kaldifst::TextNormalizer>>
LoadTextNormalizers(const std::string &rule_fsts, const std::string &rule_fars,
                    bool debug) {
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list;

  if (!rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fsts, ",", false, &files);
    tn_list.reserve(files.size());
    for (const auto &f : files) {
      if (debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("rule fst: %{public}s", f.c_str());
#else
        SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
#endif
      }
      tn_list.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
    }
  }

  if (!rule_fars.empty()) {
    if (debug) {
      SHERPA_ONNX_LOGE("Loading FST archives");
    }
    std::vector<std::string> files;
    SplitStringToVector(rule_fars, ",", false, &files);

    tn_list.reserve(files.size() + tn_list.size());

    for (const auto &f : files) {
      if (debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("rule far: %{public}s", f.c_str());
#else
        SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
#endif
      }
      std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
          fst::FarReader<fst::StdArc>::Open(f));
      for (; !reader->Done(); reader->Next()) {
        std::unique_ptr<fst::StdConstFst> r(
            fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

        tn_list.push_back(
            std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
      }
    }

    if (debug) {
      SHERPA_ONNX_LOGE("FST archives loaded!");
    }
  }

  return tn_list;
}

// Load text normalizers from rule_fsts and rule_fars (manager version for
// Android/OHOS).
template <typename Manager>
std::vector<std::unique_ptr<kaldifst::TextNormalizer>> LoadTextNormalizers(
    Manager *mgr, const std::string &rule_fsts, const std::string &rule_fars,
    bool debug) {
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list;

  if (!rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fsts, ",", false, &files);
    tn_list.reserve(files.size());
    for (const auto &f : files) {
      if (debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("rule fst: %{public}s", f.c_str());
#else
        SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
#endif
      }
      auto buf = ReadFile(mgr, f);
      std::istringstream is(std::string(buf.data(), buf.size()));
      tn_list.push_back(std::make_unique<kaldifst::TextNormalizer>(is));
    }
  }

  if (!rule_fars.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fars, ",", false, &files);
    tn_list.reserve(files.size() + tn_list.size());

    for (const auto &f : files) {
      if (debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("rule far: %{public}s", f.c_str());
#else
        SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
#endif
      }

      auto buf = ReadFile(mgr, f);

      auto fsts = ReadFstsFromFar(buf);
      for (auto &r : fsts) {
        tn_list.push_back(
            std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
      }
    }  // for (const auto &f : files)
  }  // if (!rule_fars.empty())

  return tn_list;
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TTS_TEXT_NORMALIZER_H_
