// sherpa-onnx/csrc/offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineTtsConfig::Register(ParseOptions *po) {
  model.Register(po);

  po->Register("tts-rule-fsts", &rule_fsts,
               "It not empty, it contains a list of rule FST filenames."
               "Multiple filenames are separated by a comma and they are "
               "applied from left to right. An example value: "
               "rule1.fst,rule2,fst,rule3.fst");

  po->Register(
      "tts-max-num-sentences", &max_num_sentences,
      "Maximum number of sentences that we process at a time. "
      "This is to avoid OOM for very long input text. "
      "If you set it to -1, then we process all sentences in a single batch.");
}

bool OfflineTtsConfig::Validate() const {
  if (!rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fsts, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE("Rule fst %s does not exist. ", f.c_str());
        return false;
      }
    }
  }

  return model.Validate();
}

std::string OfflineTtsConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsConfig(";
  os << "model=" << model.ToString() << ", ";
  os << "rule_fsts=\"" << rule_fsts << "\", ";
  os << "max_num_sentences=" << max_num_sentences << ")";

  return os.str();
}

OfflineTts::OfflineTts(const OfflineTtsConfig &config)
    : impl_(OfflineTtsImpl::Create(config)) {}

#if __ANDROID_API__ >= 9
OfflineTts::OfflineTts(AAssetManager *mgr, const OfflineTtsConfig &config)
    : impl_(OfflineTtsImpl::Create(mgr, config)) {}
#endif

OfflineTts::~OfflineTts() = default;

GeneratedAudio OfflineTts::Generate(
    const std::string &text, int64_t sid /*=0*/, float speed /*= 1.0*/,
    GeneratedAudioCallback callback /*= nullptr*/) const {
  return impl_->Generate(text, sid, speed, callback);
}

int32_t OfflineTts::SampleRate() const { return impl_->SampleRate(); }

int32_t OfflineTts::NumSpeakers() const { return impl_->NumSpeakers(); }

}  // namespace sherpa_onnx
