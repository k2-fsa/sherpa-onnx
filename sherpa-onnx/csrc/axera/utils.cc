// sherpa-onnx/csrc/axera/utils.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axera/utils.h"

#include <string.h>

#include <sstream>
#include <string>
#include <utility>

#include "ax_engine_api.h"   // NOLINT
#include "ax_engine_type.h"  // NOLINT
#include "ax_sys_api.h"      // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

#define SHERPA_ONNX_TO_STRING(type) \
  case type:                        \
    return #type

namespace sherpa_onnx {

static constexpr int32_t kCmnAlignSize = 128;
static const char *kSherpaOnnxAxeraSessionName = "sherpa-onnx-axera";

static std::string VectorToString(AX_S32 *arr, AX_U8 n) {
  std::ostringstream os;
  std::string sep;
  os << "[";
  for (AX_U8 i = 0; i < n; ++i) {
    os << sep << arr[i];
    sep = ", ";
  }
  os << "]";

  return os.str();
}

static const char *AxEngineDataTypeToString(AX_ENGINE_DATA_TYPE_T type) {
  switch (type) {
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_UNKNOWN);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_UINT8);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_UINT16);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_FLOAT32);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_SINT16);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_SINT8);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_SINT32);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_UINT32);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_FLOAT64);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_UINT10_PACKED);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_UINT12_PACKED);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_UINT14_PACKED);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_DT_UINT16_PACKED);
    default:
      return "Unknown data type";
  }
}

static const char *AxEngineTensorLayoutToString(
    AX_ENGINE_TENSOR_LAYOUT_T layout) {
  switch (layout) {
    SHERPA_ONNX_TO_STRING(AX_ENGINE_TENSOR_LAYOUT_UNKNOWN);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_TENSOR_LAYOUT_NHWC);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_TENSOR_LAYOUT_NCHW);
    default:
      return "Unknown data layout";
  }
}

static const char *AxEngineMemoryTypeToString(AX_ENGINE_MEMORY_TYPE_T type) {
  switch (type) {
    SHERPA_ONNX_TO_STRING(AX_ENGINE_MT_PHYSICAL);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_MT_VIRTUAL);
    SHERPA_ONNX_TO_STRING(AX_ENGINE_MT_OCM);
    default:
      return "Unknown memory type";
  }
}

/*
num_inputs: 2
num_outputs: 1
max_bach_size: 1
dynamic_bach_size: false
---input 0---
 name: x
 shape: [1, 167, 560]
 layout: AX_ENGINE_TENSOR_LAYOUT_NCHW
 memory_type: AX_ENGINE_MT_PHYSICAL
 data_type: AX_ENGINE_DT_FLOAT32
 n_size (number of bytes): 374080
---input 1---
 name: prompt
 shape: [4]
 layout: AX_ENGINE_TENSOR_LAYOUT_NCHW
 memory_type: AX_ENGINE_MT_PHYSICAL
 data_type: AX_ENGINE_DT_SINT32
 n_size (number of bytes): 16

---output 0---
 name: logits
 shape: [1, 171, 25055]
 layout: AX_ENGINE_TENSOR_LAYOUT_UNKNOWN
 memory_type: AX_ENGINE_MT_PHYSICAL
 data_type: AX_ENGINE_DT_FLOAT32
 n_size: 17137620
 */
static std::string ToString(const AX_ENGINE_IO_INFO_T *io_info) {
  std::ostringstream os;
  os << "num_inputs: " << io_info->nInputSize << "\n";
  os << "num_outputs: " << io_info->nOutputSize << "\n";
  os << "max_bach_size: " << io_info->nMaxBatchSize << "\n";
  os << "dynamic_bach_size: " << (io_info->bDynamicBatchSize ? "true" : "false")
     << "\n";

  for (AX_U32 i = 0; i < io_info->nInputSize; ++i) {
    const auto &input = io_info->pInputs[i];
    os << "---input " << i << "---\n";
    os << " name: " << input.pName << "\n";
    os << " shape: " << VectorToString(input.pShape, input.nShapeSize) << "\n";
    os << " layout: " << AxEngineTensorLayoutToString(input.eLayout) << "\n";
    os << " memory_type: " << AxEngineMemoryTypeToString(input.eMemoryType)
       << "\n";
    os << " data_type: " << AxEngineDataTypeToString(input.eDataType) << "\n";
    os << " n_size (number of bytes): " << input.nSize << "\n";
  }
  os << "\n";

  for (AX_U32 i = 0; i < io_info->nOutputSize; ++i) {
    const auto &output = io_info->pOutputs[i];
    os << "---output " << i << "---\n";
    os << " name: " << output.pName << "\n";
    os << " shape: " << VectorToString(output.pShape, output.nShapeSize)
       << "\n";
    os << " layout: " << AxEngineTensorLayoutToString(output.eLayout) << "\n";
    os << " memory_type: " << AxEngineMemoryTypeToString(output.eMemoryType)
       << "\n";
    os << " data_type: " << AxEngineDataTypeToString(output.eDataType) << "\n";
    os << " n_size: " << output.nSize << "\n";
  }

  return os.str();
}

void InitContext(const void *model_data, size_t model_data_length, bool debug,
                 AX_ENGINE_HANDLE *handle) {
  if (!handle) {
    SHERPA_ONNX_LOGE("InitContext: handle is null");
    SHERPA_ONNX_EXIT(-1);
  }

  auto ret = AX_ENGINE_CreateHandle(handle, model_data, model_data_length);
  if (ret != 0) {
    SHERPA_ONNX_LOGE("AX_ENGINE_CreateHandle failed, ret = %d", ret);
    SHERPA_ONNX_EXIT(-1);
  }

  if (debug) {
    SHERPA_ONNX_LOGE("AX_ENGINE_CreateHandle done. handle = %p", *handle);
  }

  ret = AX_ENGINE_CreateContext(*handle);
  if (ret != 0) {
    SHERPA_ONNX_LOGE("AX_ENGINE_CreateContext failed, ret = %d", ret);
    SHERPA_ONNX_EXIT(-1);
  }

  if (debug) {
    SHERPA_ONNX_LOGE("AX_ENGINE_CreateContext done.");
  }
}

void InitInputOutputAttrs(AX_ENGINE_HANDLE handle, bool debug,
                          AX_ENGINE_IO_INFO_T **io_info) {
  if (!io_info) {
    SHERPA_ONNX_LOGE("InitInputOutputAttrs: io_info is null");
    SHERPA_ONNX_EXIT(-1);
  }

  // Note(fangjun): No need to free *io_info
  auto ret = AX_ENGINE_GetIOInfo(handle, io_info);
  if (ret != 0) {
    SHERPA_ONNX_LOGE("AX_ENGINE_GetIOInfo failed, ret = %d", ret);
    SHERPA_ONNX_EXIT(-1);
  }

  if (debug) {
    SHERPA_ONNX_LOGE("AX_ENGINE_GetIOInfo done.");
    SHERPA_ONNX_LOGE("IO_INFO:\n%s", ToString(*io_info).c_str());
  }
}

void PrepareIO(AX_ENGINE_IO_INFO_T *io_info, AX_ENGINE_IO_T *io_data,
               bool debug) {
  if (!io_info || !io_data) {
    SHERPA_ONNX_LOGE("PrepareIO: io_info or io_data is null");
    SHERPA_ONNX_EXIT(-1);
  }

  memset(io_data, 0, sizeof(AX_ENGINE_IO_T));

  io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[io_info->nInputSize];

  memset(io_data->pInputs, 0,
         sizeof(AX_ENGINE_IO_BUFFER_T) * io_info->nInputSize);

  io_data->nInputSize = io_info->nInputSize;

  for (AX_U32 i = 0; i < io_info->nInputSize; ++i) {
    const auto &input = io_info->pInputs[i];
    auto &buffer = io_data->pInputs[i];

    buffer.nSize = input.nSize;

    auto ret = AX_SYS_MemAlloc(
        reinterpret_cast<AX_U64 *>(&buffer.phyAddr), &buffer.pVirAddr,
        input.nSize, kCmnAlignSize,
        reinterpret_cast<const AX_S8 *>(kSherpaOnnxAxeraSessionName));

    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to allocate memory for Input %d",
                       static_cast<int32_t>(i));
      SHERPA_ONNX_EXIT(-1);
    }
  }

  io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[io_info->nOutputSize];

  memset(io_data->pOutputs, 0,
         sizeof(AX_ENGINE_IO_BUFFER_T) * io_info->nOutputSize);

  io_data->nOutputSize = io_info->nOutputSize;

  for (AX_U32 i = 0; i < io_info->nOutputSize; ++i) {
    const auto &output = io_info->pOutputs[i];
    auto &buffer = io_data->pOutputs[i];
    buffer.nSize = output.nSize;
    auto ret = AX_SYS_MemAllocCached(
        reinterpret_cast<AX_U64 *>(&buffer.phyAddr), &buffer.pVirAddr,
        output.nSize, kCmnAlignSize,
        reinterpret_cast<const AX_S8 *>(kSherpaOnnxAxeraSessionName));

    if (ret != 0) {
      SHERPA_ONNX_LOGE("Failed to allocate memory for Output %d",
                       static_cast<int32_t>(i));
      SHERPA_ONNX_EXIT(-1);
    }
  }
}

void FreeIO(AX_ENGINE_IO_T *io_data) {
  for (AX_U32 i = 0; i < io_data->nInputSize; ++i) {
    auto &buf = io_data->pInputs[i];
    AX_SYS_MemFree(buf.phyAddr, buf.pVirAddr);
  }

  for (AX_U32 i = 0; i < io_data->nOutputSize; ++i) {
    auto &buf = io_data->pOutputs[i];
    AX_SYS_MemFree(buf.phyAddr, buf.pVirAddr);
  }
  delete[] io_data->pInputs;
  delete[] io_data->pOutputs;
}

}  // namespace sherpa_onnx
