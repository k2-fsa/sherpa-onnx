// sherpa-onnx/csrc/online-t-one-ctc-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-t-one-ctc-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineToneCtcModelConfig::Register(ParseOptions *po) {
  po->Register("t-one-ctc-model", &model,
               "Path to CTC model.onnx from T-one. Please see "
               "https://github.com/k2-fsa/sherpa-onnx/pull/2571");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOnlineToneCtcModelConfig(
    const OnlineToneCtcModelConfig &config, const Exists &exists) {
  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("T-one CTC model '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  return true;
}

bool OnlineToneCtcModelConfig::Validate() const {
  return ValidateOnlineToneCtcModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OnlineToneCtcModelConfig::Validate(Manager *mgr) const {
  return ValidateOnlineToneCtcModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OnlineToneCtcModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OnlineToneCtcModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OnlineToneCtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineToneCtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
