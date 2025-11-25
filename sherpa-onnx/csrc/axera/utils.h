// sherpa-onnx/csrc/axera/utils.h
//
// Copyright      2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_AXERA_UTILS_H_
#define SHERPA_ONNX_CSRC_AXERA_UTILS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "ax_engine_api.h"

namespace sherpa_onnx {

void ConvertNCHWtoNHWC(const float *src, int32_t n, int32_t channel,
                       int32_t height, int32_t width, float *dst);

std::string ToString(const AX_ENGINE_IO_INFO_T *io_info);

std::unordered_map<std::string, std::string> Parse(const char *custom_string,
                                                   bool debug = false);

void InitEngine(bool debug);

void InitContext(void *model_data, size_t model_data_length, bool debug,
                 AX_ENGINE_HANDLE *handle);

void InitInputOutputAttrs(AX_ENGINE_HANDLE handle, bool debug,
                          AX_ENGINE_IO_INFO_T **io_info);

void PrepareIO(AX_ENGINE_IO_INFO_T *io_info, AX_ENGINE_IO_T *io_data,
               bool debug);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_UTILS_H_
