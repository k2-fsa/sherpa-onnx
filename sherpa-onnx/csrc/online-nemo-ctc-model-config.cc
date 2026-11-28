// sherpa-onnx/csrc/online-nemo-ctc-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-nemo-ctc-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineNeMoCtcModelConfig::Register(ParseOptions *po) {
  po->Register("nemo-ctc-model", &model,
               "Path to CTC model.onnx from NeMo. Please see "
               "https://github.com/k2-fsa/sherpa-onnx/pull/843");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOnlineNeMoCtcModelConfig(
    const OnlineNeMoCtcModelConfig &config, const Exists &exists) {
  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("NeMo CTC model '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  return true;
}

bool OnlineNeMoCtcModelConfig::Validate() const {
  return ValidateOnlineNeMoCtcModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OnlineNeMoCtcModelConfig::Validate(Manager *mgr) const {
  return ValidateOnlineNeMoCtcModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OnlineNeMoCtcModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OnlineNeMoCtcModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OnlineNeMoCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineNeMoCtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
