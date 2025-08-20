// sherpa-onnx/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VERSION_H_
#define SHERPA_ONNX_CSRC_VERSION_H_

namespace sherpa_onnx {

// Please don't free the returned pointer.
// Please don't modify the memory pointed by the returned pointer.
//
// The memory pointed by the returned pointer is statically allocated.
const char *GetVersionStr();

// Please don't free the returned pointer.
// Please don't modify the memory pointed by the returned pointer.
//
// The memory pointed by the returned pointer is statically allocated.
const char *GetGitSha1();

// Please don't free the returned pointer.
// Please don't modify the memory pointed by the returned pointer.
//
// The memory pointed by the returned pointer is statically allocated.
const char *GetGitDate();

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VERSION_H_
