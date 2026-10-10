// sherpa-onnx/csrc/offline-tts-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-impl.h"

#include <memory>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/offline-tts-kitten-impl.h"
#include "sherpa-onnx/csrc/offline-tts-kokoro-impl.h"
#include "sherpa-onnx/csrc/offline-tts-matcha-impl.h"
#include "sherpa-onnx/csrc/offline-tts-pocket-impl.h"
#include "sherpa-onnx/csrc/offline-tts-supertonic-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-impl.h"
#include "sherpa-onnx/csrc/offline-tts-zipvoice-impl.h"

#ifdef SHERPA_ONNX_ENABLE_AXCL
#include "sherpa-onnx/csrc/axcl/offline-tts-kokoro-axcl-impl.h"
#endif

namespace sherpa_onnx {

std::vector<int64_t> OfflineTtsImpl::AddBlank(const std::vector<int64_t> &x,
                                              int32_t blank_id /*= 0*/) const {
  // we assume the blank ID is 0
  std::vector<int64_t> buffer(x.size() * 2 + 1, blank_id);
  int32_t i = 1;
  for (auto k : x) {
    buffer[i] = k;
    i += 2;
  }
  return buffer;
}

std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    const OfflineTtsConfig &config) {
  if (config.model.provider == "axcl") {
#if SHERPA_ONNX_ENABLE_AXCL
    if (!config.model.kokoro.model.empty()) {
      return std::make_unique<OfflineTtsKokoroAxclImpl<OfflineTtsKokoroModelAxcl>>(
          config);
    } else {
      SHERPA_ONNX_LOGE(
          "Only Kokoro models are currently supported by axcl for "
          "non-streaming TTS.");
      SHERPA_ONNX_EXIT(-1);
      return nullptr;
    }

#else
    SHERPA_ONNX_LOGE(
        "Please rebuild sherpa-onnx with -DSHERPA_ONNX_ENABLE_AXCL=ON if you "
        "want to use axcl. See also "
        "https://k2-fsa.github.io/sherpa/onnx/axcl/install.html");
    SHERPA_ONNX_EXIT(-1);
    return nullptr;
#endif
  }

  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(config);
  } else if (!config.model.matcha.acoustic_model.empty()) {
    return std::make_unique<OfflineTtsMatchaImpl>(config);
  } else if (!config.model.zipvoice.encoder.empty() &&
             !config.model.zipvoice.decoder.empty()) {
    return std::make_unique<OfflineTtsZipvoiceImpl>(config);
  } else if (!config.model.kokoro.model.empty()) {
    return std::make_unique<OfflineTtsKokoroImpl>(config);
  } else if (!config.model.kitten.model.empty()) {
    return std::make_unique<OfflineTtsKittenImpl>(config);
  } else if (!config.model.pocket.lm_flow.empty()) {
    return std::make_unique<OfflineTtsPocketImpl>(config);
  } else if (!config.model.supertonic.tts_json.empty()) {
    return std::make_unique<OfflineTtsSupertonicImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please provide a tts model.");

  return {};
}

template <typename Manager>
std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    Manager *mgr, const OfflineTtsConfig &config) {
  if (!config.model.vits.model.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(mgr, config);
  } else if (!config.model.matcha.acoustic_model.empty()) {
    return std::make_unique<OfflineTtsMatchaImpl>(mgr, config);
  } else if (!config.model.zipvoice.encoder.empty() &&
             !config.model.zipvoice.decoder.empty()) {
    return std::make_unique<OfflineTtsZipvoiceImpl>(mgr, config);
  } else if (!config.model.kokoro.model.empty()) {
    return std::make_unique<OfflineTtsKokoroImpl>(mgr, config);
  } else if (!config.model.kitten.model.empty()) {
    return std::make_unique<OfflineTtsKittenImpl>(mgr, config);
  } else if (!config.model.pocket.lm_flow.empty()) {
    return std::make_unique<OfflineTtsPocketImpl>(mgr, config);
  } else if (!config.model.supertonic.tts_json.empty()) {
    return std::make_unique<OfflineTtsSupertonicImpl>(mgr, config);
  }

  SHERPA_ONNX_LOGE("Please provide a tts model.");
  return {};
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    AAssetManager *mgr, const OfflineTtsConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    NativeResourceManager *mgr, const OfflineTtsConfig &config);
#endif

}  // namespace sherpa_onnx
