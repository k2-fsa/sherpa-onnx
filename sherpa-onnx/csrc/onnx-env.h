// sherpa-onnx/csrc/onnx-env.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONNX_ENV_H_
#define SHERPA_ONNX_CSRC_ONNX_ENV_H_

#include <cstdint>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

// Returns a process-wide Ort::Env created via CreateEnvWithGlobalThreadPools.
//
// ONNX Runtime builds where per-session threads are disabled at compile time
// (notably the wasm-with-pthreads build, where DEFAULT_USE_PER_SESSION_THREADS
// is forced to false) require every Ort::Session to be constructed with an
// Env initialized through global thread pools. Constructing a session with a
// plain Env (CreateEnv) trips an assertion inside InferenceSession.
//
// The first caller's num_threads value (clamped to >= 1) determines the
// global thread-pool size; subsequent calls reuse the same Env.
Ort::Env &GetGlobalOrtEnv(int32_t num_threads = 1);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONNX_ENV_H_
