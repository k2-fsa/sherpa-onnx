// sherpa-onnx/csrc/ascend/offline-sense-voice-model-ascend.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/ascend/offline-sense-voice-model-ascend.h"

#include "sherpa-onnx/csrc/ascend/macros.h"
#include "sherpa-onnx/csrc/ascend/utils.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelAscend::Impl {
 public:
  Impl() {
    std::string filename = "./model.om";
    model_ = std::make_unique<AclModel>(filename);
    auto s = model_->GetInfo();
    SHERPA_ONNX_LOGE("%s", s.c_str());
  }

 private:
  std::unique_ptr<AclModel> model_;
};

OfflineSenseVoiceModelAscend::OfflineSenseVoiceModelAscend()
    : impl_(std::make_unique<Impl>()) {}

OfflineSenseVoiceModelAscend::~OfflineSenseVoiceModelAscend() = default;

}  // namespace sherpa_onnx
