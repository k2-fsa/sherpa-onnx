// Copyright (c) 2026 LittleMouse
#include "sherpa-onnx/csrc/axera/tts-session.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ax_sys_api.h"  // NOLINT
#include "sherpa-onnx/csrc/axera/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"

namespace sherpa_onnx {
AxeraTtsSession::AxeraTtsSession(const std::string &filename, bool debug) {
  auto data = ReadFile(filename);
  InitContext(data.data(), data.size(), debug, &handle_);
  InitInputOutputAttrs(handle_, debug, &info_);
  PrepareIO(info_, &io_, debug);
}
AxeraTtsSession::~AxeraTtsSession() {
  FreeIO(&io_);
  if (handle_) AX_ENGINE_DestroyHandle(handle_);
}
int32_t AxeraTtsSession::Index(const std::string &name, bool input) const {
  auto count = input ? info_->nInputSize : info_->nOutputSize;
  auto values = input ? info_->pInputs : info_->pOutputs;
  for (uint32_t i = 0; i < count; ++i) {
    if (name == values[i].pName) return i;
  }
  throw std::runtime_error("Missing AXERA tensor: " + name);
}
void AxeraTtsSession::Check(const std::string &name, bool input,
                            AX_ENGINE_DATA_TYPE_T dtype,
                            const std::vector<int32_t> &shape) const {
  auto i = Index(name, input);
  const auto &v = (input ? info_->pInputs : info_->pOutputs)[i];
  if (v.eDataType != dtype || v.nShapeSize != shape.size() ||
      !std::equal(shape.begin(), shape.end(), v.pShape)) {
    throw std::runtime_error("Unexpected AXERA tensor type/shape: " + name);
  }
  size_t elements = 1;
  for (auto d : shape) elements *= d;
  size_t bytes = elements * (dtype == AX_ENGINE_DT_UINT8 ? 1 : 4);
  if (bytes != v.nSize)
    throw std::runtime_error("Unexpected AXERA tensor size: " + name);
}
void AxeraTtsSession::CheckCount(uint32_t inputs, uint32_t outputs) const {
  if (info_->nInputSize != inputs || info_->nOutputSize != outputs) {
    throw std::runtime_error("Unexpected AXERA model IO count");
  }
}
void AxeraTtsSession::CheckInput(const std::string &name,
                                 AX_ENGINE_DATA_TYPE_T dtype,
                                 const std::vector<int32_t> &shape) const {
  Check(name, true, dtype, shape);
}
void AxeraTtsSession::CheckOutput(const std::string &name,
                                  AX_ENGINE_DATA_TYPE_T dtype,
                                  const std::vector<int32_t> &shape) const {
  Check(name, false, dtype, shape);
}
void AxeraTtsSession::Set(const std::string &name, const void *data,
                          size_t bytes) {
  auto i = Index(name, true);
  if (bytes != info_->pInputs[i].nSize) {
    throw std::runtime_error("AXERA input byte count mismatch: " + name);
  }
  std::memcpy(io_.pInputs[i].pVirAddr, data, bytes);
}
void AxeraTtsSession::Run() {
  auto ret = AX_ENGINE_RunSync(handle_, &io_);
  if (ret)
    throw std::runtime_error("AX_ENGINE_RunSync failed: " +
                             std::to_string(ret));
}
std::vector<float> AxeraTtsSession::Get(const std::string &name) {
  auto i = Index(name, false);
  auto &buffer = io_.pOutputs[i];
  if (info_->pOutputs[i].eDataType != AX_ENGINE_DT_FLOAT32) {
    throw std::runtime_error("AXERA output is not float32: " + name);
  }
  auto ret =
      AX_SYS_MinvalidateCache(buffer.phyAddr, buffer.pVirAddr, buffer.nSize);
  if (ret) throw std::runtime_error("AXERA output cache invalidation failed");
  std::vector<float> result(buffer.nSize / sizeof(float));
  std::memcpy(result.data(), buffer.pVirAddr, buffer.nSize);
  return result;
}
std::vector<uint8_t> AxeraTtsSession::GetBytes(const std::string &name) {
  auto i = Index(name, false);
  auto &buffer = io_.pOutputs[i];
  if (info_->pOutputs[i].eDataType != AX_ENGINE_DT_UINT8) {
    throw std::runtime_error("AXERA output is not uint8: " + name);
  }
  auto ret =
      AX_SYS_MinvalidateCache(buffer.phyAddr, buffer.pVirAddr, buffer.nSize);
  if (ret) throw std::runtime_error("AXERA output cache invalidation failed");
  std::vector<uint8_t> result(buffer.nSize);
  std::memcpy(result.data(), buffer.pVirAddr, buffer.nSize);
  return result;
}
}  // namespace sherpa_onnx
