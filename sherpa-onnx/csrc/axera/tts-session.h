// Copyright (c) 2026 LittleMouse
#ifndef SHERPA_ONNX_CSRC_AXERA_TTS_SESSION_H_
#define SHERPA_ONNX_CSRC_AXERA_TTS_SESSION_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ax_engine_api.h"  // NOLINT
#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"

namespace sherpa_onnx {
// Owns one context and its physical IO buffers. Calls are serialized by
// callers.
class AxeraTtsSession {
 public:
  explicit AxeraTtsSession(const std::string &filename, bool debug);
  ~AxeraTtsSession();
  AxeraTtsSession(const AxeraTtsSession &) = delete;
  AxeraTtsSession &operator=(const AxeraTtsSession &) = delete;
  void CheckCount(uint32_t inputs, uint32_t outputs) const;
  void CheckInput(const std::string &name, AX_ENGINE_DATA_TYPE_T dtype,
                  const std::vector<int32_t> &shape) const;
  void CheckOutput(const std::string &name, AX_ENGINE_DATA_TYPE_T dtype,
                   const std::vector<int32_t> &shape) const;
  void Set(const std::string &name, const void *data, size_t bytes);
  std::vector<float> Get(const std::string &name);
  std::vector<uint8_t> GetBytes(const std::string &name);
  void Run();

 private:
  int32_t Index(const std::string &name, bool input) const;
  void Check(const std::string &name, bool input, AX_ENGINE_DATA_TYPE_T dtype,
             const std::vector<int32_t> &shape) const;
  AxEngineGuard guard_;
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *info_ = nullptr;
  AX_ENGINE_IO_T io_{};
};
}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_AXERA_TTS_SESSION_H_
