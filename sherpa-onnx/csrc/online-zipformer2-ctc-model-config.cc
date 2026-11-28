// sherpa-onnx/csrc/online-zipformer2-ctc-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer2-ctc-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineZipformer2CtcModelConfig::Register(ParseOptions *po) {
  po->Register("zipformer2-ctc-model", &model,
               "Path to CTC model.onnx. See also "
               "https://github.com/k2-fsa/icefall/pull/1413");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOnlineZipformer2CtcModelConfig(
    const OnlineZipformer2CtcModelConfig &config, const Exists &exists) {
  if (config.model.empty()) {
    SHERPA_ONNX_LOGE("--zipformer2-ctc-model is empty!");
    return false;
  }

  if (!exists(config.model)) {
    SHERPA_ONNX_LOGE("--zipformer2-ctc-model '%s' does not exist",
                     config.model.c_str());
    return false;
  }

  return true;
}

bool OnlineZipformer2CtcModelConfig::Validate() const {
  return ValidateOnlineZipformer2CtcModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OnlineZipformer2CtcModelConfig::Validate(Manager *mgr) const {
  return ValidateOnlineZipformer2CtcModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OnlineZipformer2CtcModelConfig::Validate(
    AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OnlineZipformer2CtcModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OnlineZipformer2CtcModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineZipformer2CtcModelConfig(";
  os << "model=\"" << model << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
