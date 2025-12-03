// sherpa-onnx/csrc/axera/utils.h
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#ifndef SHERPA_ONNX_CSRC_AXERA_UTILS_H_
#define SHERPA_ONNX_CSRC_AXERA_UTILS_H_

#include <cstddef>

#include "ax_engine_api.h"

namespace sherpa_onnx {

void InitContext(const void *model_data, size_t model_data_length, bool debug,
                 AX_ENGINE_HANDLE *handle);

void InitInputOutputAttrs(AX_ENGINE_HANDLE handle, bool debug,
                          AX_ENGINE_IO_INFO_T **io_info);

void PrepareIO(AX_ENGINE_IO_INFO_T *io_info, AX_ENGINE_IO_T *io_data,
               bool debug);

void FreeIO(AX_ENGINE_IO_T *io_data);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AXERA_UTILS_H_
