// sherpa-onnx/csrc/ort-env.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ORT_ENV_H_
#define SHERPA_ONNX_CSRC_ORT_ENV_H_

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

// Create an Ort::Env with appropriate threading configuration.
// In WASM builds, onnxruntime's default thread pool creation can cause
// abort() from background pthreads. Using CreateEnvWithGlobalThreadPools
// with single-threaded pools avoids this. Per-session threading is
// configured separately via session.cc SetIntraOpNumThreads.
inline Ort::Env CreateOrtEnv() {
#if SHERPA_ONNX_ENABLE_WASM
  const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtThreadingOptions *tp = nullptr;
  (void)api->CreateThreadingOptions(&tp);
  (void)api->SetGlobalIntraOpNumThreads(tp, 1);
  (void)api->SetGlobalInterOpNumThreads(tp, 1);
  OrtEnv *env = nullptr;
  (void)api->CreateEnvWithGlobalThreadPools(ORT_LOGGING_LEVEL_ERROR,
                                            "sherpa-onnx", tp, &env);
  (void)api->ReleaseThreadingOptions(tp);
  return Ort::Env(env);
#else
  return Ort::Env(ORT_LOGGING_LEVEL_ERROR);
#endif
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ORT_ENV_H_
