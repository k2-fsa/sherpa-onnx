// sherpa-onnx/csrc/online-paraformer-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-paraformer-model-config.h"

#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OnlineParaformerModelConfig::Register(ParseOptions *po) {
  po->Register("paraformer-encoder", &encoder,
               "Path to encoder.onnx of paraformer.");
  po->Register("paraformer-decoder", &decoder,
               "Path to decoder.onnx of paraformer.");
}

// 'exists' resolves a path either against the file system or against a
// resource manager.
template <typename Exists>
static bool ValidateOnlineParaformerModelConfig(
    const OnlineParaformerModelConfig &config, const Exists &exists) {
  if (!exists(config.encoder)) {
    SHERPA_ONNX_LOGE("Paraformer encoder '%s' does not exist",
                     config.encoder.c_str());
    return false;
  }

  if (!exists(config.decoder)) {
    SHERPA_ONNX_LOGE("Paraformer decoder '%s' does not exist",
                     config.decoder.c_str());
    return false;
  }

  return true;
}

bool OnlineParaformerModelConfig::Validate() const {
  return ValidateOnlineParaformerModelConfig(
      *this, [](const std::string &p) -> bool { return FileExists(p); });
}

template <typename Manager>
bool OnlineParaformerModelConfig::Validate(Manager *mgr) const {
  return ValidateOnlineParaformerModelConfig(
      *this,
      [mgr](const std::string &p) -> bool { return FileExists(mgr, p); });
}

#if __ANDROID_API__ >= 9
template bool OnlineParaformerModelConfig::Validate(AAssetManager *mgr) const;
#endif

#if __OHOS__
template bool OnlineParaformerModelConfig::Validate(
    NativeResourceManager *mgr) const;
#endif

std::string OnlineParaformerModelConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineParaformerModelConfig(";
  os << "encoder=\"" << encoder << "\", ";
  os << "decoder=\"" << decoder << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
