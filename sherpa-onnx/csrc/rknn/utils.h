// sherpa-onnx/csrc/utils.h
//
// Copyright      2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_RKNN_UTILS_H_
#define SHERPA_ONNX_CSRC_RKNN_UTILS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "rknn_api.h"  // NOLINT

namespace sherpa_onnx {

void ConvertNCHWtoNHWC(const float *src, int32_t n, int32_t channel,
                       int32_t height, int32_t width, float *dst);

std::string ToString(const rknn_tensor_attr &attr);

std::unordered_map<std::string, std::string> Parse(
    const rknn_custom_string &custom_string, bool debug = false);

void InitContext(void *model_data, size_t model_data_length, bool debug,
                 rknn_context *ctx);

void InitInputOutputAttrs(rknn_context ctx, bool debug,
                          std::vector<rknn_tensor_attr> *input_attrs,
                          std::vector<rknn_tensor_attr> *output_attrs);

rknn_custom_string GetCustomString(rknn_context ctx, bool debug);

void SetCoreMask(rknn_context ctx, int32_t num_threads);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RKNN_UTILS_H_
