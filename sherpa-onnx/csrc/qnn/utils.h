// sherpa-onnx/csrc/qnn/utils.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_QNN_UTILS_H_
#define SHERPA_ONNX_CSRC_QNN_UTILS_H_
#include <stdio.h>

#include <cstdint>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

void PrintTensor(Qnn_TensorV2_t t);

template <typename T>
std::vector<T> ReadFile(const std::string &filename) {
  FILE *fp = fopen(filename.c_str(), "rb");
  fseek(fp, 0, SEEK_END);
  int32_t n = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  std::vector<T> ans(n / sizeof(T));
  fread(ans.data(), sizeof(T), n, fp);
  fclose(fp);

  return ans;
}

// float -> uint16_t
void FillData(Qnn_Tensor_t *t, const float *data, int32_t n);

// int32_t -> int32_t
void FillData(Qnn_Tensor_t *t, const int32_t *data, int32_t n);

// uint16_t -> float
void GetData(const Qnn_Tensor_t *t, float *data, int32_t n);

void FreeTensor(Qnn_Tensor_t *t);

using TensorPtr = std::unique_ptr<Qnn_Tensor_t, decltype(&FreeTensor)>;

void CopyTensorInfo(const Qnn_Tensor_t &src, Qnn_Tensor_t &dst);

std::string QuantizationEncodingToString(Qnn_QuantizationEncoding_t q);

std::string TensorDataTypeToString(Qnn_DataType_t t);

using QnnInterfaceGetProvidersFnType = Qnn_ErrorHandle_t (*)(
    const QnnInterface_t ***provider_list, uint32_t *num_providers);

using QnnSystemInterfaceGetProvidersFnType = Qnn_ErrorHandle_t (*)(
    const QnnSystemInterface_t ***provider_list, uint32_t *num_providers);

struct GraphInfo {
  Qnn_GraphHandle_t graph;
  char *graph_name;
  Qnn_Tensor_t *input_tensors;
  uint32_t num_input_tensors;
  Qnn_Tensor_t *output_tensors;
  uint32_t num_output_tensors;
};

struct GraphConfigInfo {
  char *graph_name;
  const QnnGraph_Config_t **graph_configs;
};

using ComposeGraphsFnHandleType = Qnn_ErrorHandle_t (*)(
    Qnn_BackendHandle_t backend_handle, QNN_INTERFACE_VER_TYPE interface,
    Qnn_ContextHandle_t context_handle,
    const GraphConfigInfo **graphs_config_info,
    const uint32_t num_graphs_config_info, GraphInfo ***graphs_info,
    uint32_t *num_graphs_info, bool debug, QnnLog_Callback_t logCallback,
    QnnLog_Level_t max_log_level);

using FreeGraphInfoFnHandleType =
    Qnn_ErrorHandle_t (*)(GraphInfo ***, uint32_t num_graphs_info);

void LogCallback(const char *fmt, QnnLog_Level_t level, uint64_t timestamp,
                 va_list args);

bool CopyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binary_info,
                              GraphInfo **&graphs_info, uint32_t &graphs_count);
#endif  // SHERPA_ONNX_CSRC_QNN_UTILS_H_
