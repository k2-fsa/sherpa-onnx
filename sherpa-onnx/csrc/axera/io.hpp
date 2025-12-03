// sherpa-onnx/csrc/axera/io.hpp
//
// This file is adapted from AXERA's ax-samples project.
// See the original BSD 3-Clause license below.
//
// Copyright (c)  2025  M5Stack Technology CO LTD

/*
 * AXERA is pleased to support the open source community by making ax-samples
 * available.
 *
 * Copyright (c) 2022, AXERA Semiconductor (Shanghai) Co., Ltd. All rights
 * reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
/*
 * Author: AXERA Corporation
 */

#pragma once

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#define AX_CMM_ALIGN_SIZE 128

inline const char *AX_CMM_SESSION_NAME = "ax-samples-cmm";

typedef enum {
  AX_ENGINE_ABST_DEFAULT = 0,
  AX_ENGINE_ABST_CACHED = 1,
} AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

typedef std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T,
                  AX_ENGINE_ALLOC_BUFFER_STRATEGY_T>
    INPUT_OUTPUT_ALLOC_STRATEGY;

#define SAMPLE_AX_ENGINE_DEAL_HANDLE        \
  if (0 != ret) {                           \
    return AX_ENGINE_DestroyHandle(handle); \
  }

#define SAMPLE_AX_ENGINE_DEAL_HANDLE_IO     \
  if (0 != ret) {                           \
    middleware::free_io(&io_data);          \
    return AX_ENGINE_DestroyHandle(handle); \
  }

namespace middleware {

inline void free_io_index(AX_ENGINE_IO_BUFFER_T *io_buf, size_t index) {
  for (int i = 0; i < (int)index; ++i) {
    AX_ENGINE_IO_BUFFER_T *pBuf = io_buf + i;
    AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
  }
}

inline void free_io(AX_ENGINE_IO_T *io) {
  for (size_t j = 0; j < io->nInputSize; ++j) {
    AX_ENGINE_IO_BUFFER_T *pBuf = io->pInputs + j;
    AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
  }
  for (size_t j = 0; j < io->nOutputSize; ++j) {
    AX_ENGINE_IO_BUFFER_T *pBuf = io->pOutputs + j;
    AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
  }
  delete[] io->pInputs;
  delete[] io->pOutputs;
}

static inline int prepare_io(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data,
                             INPUT_OUTPUT_ALLOC_STRATEGY strategy) {
  memset(io_data, 0, sizeof(*io_data));
  io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
  memset(io_data->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);
  io_data->nInputSize = info->nInputSize;

  auto ret = 0;
  for (int i = 0; i < (int)info->nInputSize; ++i) {
    auto meta = info->pInputs[i];
    auto buffer = &io_data->pInputs[i];
    if (strategy.first == AX_ENGINE_ABST_CACHED) {
      ret = AX_SYS_MemAllocCached(
          (AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize,
          AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
    } else {
      ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr,
                            meta.nSize, AX_CMM_ALIGN_SIZE,
                            (const AX_S8 *)(AX_CMM_SESSION_NAME));
    }

    if (ret != 0) {
      free_io_index(io_data->pInputs, i);
      fprintf(
          stderr,
          "Allocate input{%d} { phy: %p, vir: %p, size: %lu Bytes }. fail \n",
          i, (void *)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
      return ret;
    }
    // fprintf(stderr, "Allocate input{%d} { phy: %p, vir: %p, size: %lu Bytes
    // }. \n", i, (void*)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
  }

  io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
  memset(io_data->pOutputs, 0,
         sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);
  io_data->nOutputSize = info->nOutputSize;
  for (int i = 0; i < (int)info->nOutputSize; ++i) {
    auto meta = info->pOutputs[i];
    auto buffer = &io_data->pOutputs[i];
    buffer->nSize = meta.nSize;
    if (strategy.second == AX_ENGINE_ABST_CACHED) {
      ret = AX_SYS_MemAllocCached(
          (AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize,
          AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
    } else {
      ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr,
                            meta.nSize, AX_CMM_ALIGN_SIZE,
                            (const AX_S8 *)(AX_CMM_SESSION_NAME));
    }
    if (ret != 0) {
      fprintf(
          stderr,
          "Allocate output{%d} { phy: %p, vir: %p, size: %lu Bytes }. fail \n",
          i, (void *)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
      free_io_index(io_data->pInputs, io_data->nInputSize);
      free_io_index(io_data->pOutputs, i);
      return ret;
    }
    // fprintf(stderr, "Allocate output{%d} { phy: %p, vir: %p, size: %lu Bytes
    // }.\n", i, (void*)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
  }

  return 0;
}

static int push_input(const std::vector<uint8_t> &data, AX_ENGINE_IO_T *io_t,
                      AX_ENGINE_IO_INFO_T *info_t) {
  if (info_t->nInputSize != 1) {
    fprintf(stderr, "Only support Input size == 1 current now");
    return -1;
  }

  if (data.size() != info_t->pInputs[0].nSize) {
    fprintf(stderr,
            "The input data size is not matched with tensor {name: %s, size: "
            "%d}.\n",
            info_t->pInputs[0].pName, info_t->pInputs[0].nSize);
    return -1;
  }

  memcpy(io_t->pInputs[0].pVirAddr, data.data(), data.size());

  return 0;
}

static void print_io_info(AX_ENGINE_IO_INFO_T *io_info) {
  static std::map<AX_ENGINE_DATA_TYPE_T, const char *> data_type = {
      {AX_ENGINE_DT_UNKNOWN, "UNKNOWN"},
      {AX_ENGINE_DT_UINT8, "UINT8"},
      {AX_ENGINE_DT_UINT16, "UINT16"},
      {AX_ENGINE_DT_FLOAT32, "FLOAT32"},
      {AX_ENGINE_DT_SINT16, "SINT16"},
      {AX_ENGINE_DT_SINT8, "SINT8"},
      {AX_ENGINE_DT_SINT32, "SINT32"},
      {AX_ENGINE_DT_UINT32, "UINT32"},
      {AX_ENGINE_DT_FLOAT64, "FLOAT64"},
      {AX_ENGINE_DT_UINT10_PACKED, "UINT10_PACKED"},
      {AX_ENGINE_DT_UINT12_PACKED, "UINT12_PACKED"},
      {AX_ENGINE_DT_UINT14_PACKED, "UINT14_PACKED"},
      {AX_ENGINE_DT_UINT16_PACKED, "UINT16_PACKED"},
  };

  static std::map<AX_ENGINE_COLOR_SPACE_T, const char *> color_type = {
      {AX_ENGINE_CS_FEATUREMAP, "FEATUREMAP"},
      {AX_ENGINE_CS_RAW8, "RAW8"},
      {AX_ENGINE_CS_RAW10, "RAW10"},
      {AX_ENGINE_CS_RAW12, "RAW12"},
      {AX_ENGINE_CS_RAW14, "RAW14"},
      {AX_ENGINE_CS_RAW16, "RAW16"},
      {AX_ENGINE_CS_NV12, "NV12"},
      {AX_ENGINE_CS_NV21, "NV21"},
      {AX_ENGINE_CS_RGB, "RGB"},
      {AX_ENGINE_CS_BGR, "BGR"},
      {AX_ENGINE_CS_RGBA, "RGBA"},
      {AX_ENGINE_CS_GRAY, "GRAY"},
      {AX_ENGINE_CS_YUV444, "YUV444"},
  };
  printf("\ninput size: %d\n", io_info->nInputSize);
  for (uint32_t i = 0; i < io_info->nInputSize; ++i) {
    // print shape info,like [batchsize x channel x height x width]
    auto &info = io_info->pInputs[i];
    printf("    name: \e[1;32m%8s", info.pName);

    std::string dt = "unknown";
    if (data_type.find(info.eDataType) != data_type.end()) {
      dt = data_type[info.eDataType];
      printf(" \e[1;34m[%s] ", dt.c_str());
    } else {
      printf(" \e[1;31m[%s] ", dt.c_str());
    }

    std::string ct = "unknown";
    if (info.pExtraMeta &&
        color_type.find(info.pExtraMeta->eColorSpace) != color_type.end()) {
      ct = color_type[info.pExtraMeta->eColorSpace];
      printf("\e[1;34m[%s]", ct.c_str());
    } else {
      printf("\e[1;31m[%s]", ct.c_str());
    }
    printf(" \n        \e[1;31m");

    for (AX_U8 s = 0; s < info.nShapeSize; s++) {
      printf("%d", info.pShape[s]);
      if (s != info.nShapeSize - 1) {
        printf(" x ");
      }
    }
    printf("\e[0m\n\n");
  }

  printf("\noutput size: %d\n", io_info->nOutputSize);
  for (uint32_t i = 0; i < io_info->nOutputSize; ++i) {
    // print shape info,like [batchsize x channel x height x width]
    auto &info = io_info->pOutputs[i];
    printf("    name: \e[1;32m%8s \e[1;34m[%s]\e[0m\n        \e[1;31m",
           info.pName, data_type[info.eDataType]);
    for (AX_U8 s = 0; s < info.nShapeSize; s++) {
      printf("%d", info.pShape[s]);
      if (s != info.nShapeSize - 1) {
        printf(" x ");
      }
    }
    printf("\e[0m\n\n");
  }
}

}  // namespace middleware
