// sherpa-onnx/csrc/utils.cc
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/utils.h"

#include <string.h>

#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/rknn/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void ConvertNCHWtoNHWC(const float *src, int32_t n, int32_t channel,
                       int32_t height, int32_t width, float *dst) {
  for (int32_t i = 0; i < n; ++i) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t w = 0; w < width; ++w) {
        for (int32_t c = 0; c < channel; ++c) {
          // dst[h, w, c] = src[c, h, w]
          dst[i * height * width * channel + h * width * channel + w * channel +
              c] = src[i * height * width * channel + c * height * width +
                       h * width + w];
        }
      }
    }
  }
}

std::string ToString(const rknn_tensor_attr &attr) {
  std::ostringstream os;
  os << "{";
  os << attr.index;
  os << ", name: " << attr.name;
  os << ", shape: (";
  std::string sep;
  for (int32_t i = 0; i < static_cast<int32_t>(attr.n_dims); ++i) {
    os << sep << attr.dims[i];
    sep = ",";
  }
  os << ")";
  os << ", n_elems: " << attr.n_elems;
  os << ", size: " << attr.size;
  os << ", fmt: " << get_format_string(attr.fmt);
  os << ", type: " << get_type_string(attr.type);
  os << ", pass_through: " << (attr.pass_through ? "true" : "false");
  os << "}";
  return os.str();
}

std::unordered_map<std::string, std::string> Parse(
    const rknn_custom_string &custom_string, bool debug /*= false*/) {
  std::unordered_map<std::string, std::string> ans;
  std::vector<std::string> fields;
  SplitStringToVector(custom_string.string, ";", false, &fields);

  std::vector<std::string> tmp;
  for (const auto &f : fields) {
    SplitStringToVector(f, "=", false, &tmp);
    if (tmp.size() != 2) {
      SHERPA_ONNX_LOGE("Invalid custom string %s for %s", custom_string.string,
                       f.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    ans[std::move(tmp[0])] = std::move(tmp[1]);
  }

  if (debug) {
    for (const auto &p : ans) {
      SHERPA_ONNX_LOGE("%s: %s", p.first.c_str(), p.second.c_str());
    }
  }

  return ans;
}

void InitContext(void *model_data, size_t model_data_length, bool debug,
                 rknn_context *ctx) {
  auto ret = rknn_init(ctx, model_data, model_data_length, 0, nullptr);
  SHERPA_ONNX_RKNN_CHECK(ret, "Failed to init rknn");

  if (debug) {
    rknn_sdk_version v;
    ret = rknn_query(*ctx, RKNN_QUERY_SDK_VERSION, &v, sizeof(v));
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get rknn sdk version");

    SHERPA_ONNX_LOGE("sdk api version: %s, driver version: %s", v.api_version,
                     v.drv_version);
  }
}

void InitInputOutputAttrs(rknn_context ctx, bool debug,
                          std::vector<rknn_tensor_attr> *input_attrs,
                          std::vector<rknn_tensor_attr> *output_attrs) {
  rknn_input_output_num io_num;
  auto ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get I/O information for the model");

  if (debug) {
    SHERPA_ONNX_LOGE("model: %d inputs, %d outputs",
                     static_cast<int32_t>(io_num.n_input),
                     static_cast<int32_t>(io_num.n_output));
  }

  input_attrs->resize(io_num.n_input);
  output_attrs->resize(io_num.n_output);

  int32_t i = 0;
  for (auto &attr : *input_attrs) {
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for model input %d", i);
    i += 1;
  }

  if (debug) {
    std::ostringstream os;
    std::string sep;
    for (auto &attr : *input_attrs) {
      os << sep << ToString(attr);
      sep = "\n";
    }
    SHERPA_ONNX_LOGE("\n----------Model inputs info----------\n%s",
                     os.str().c_str());
  }

  i = 0;
  for (auto &attr : *output_attrs) {
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for model output %d", i);
    i += 1;
  }

  if (debug) {
    std::ostringstream os;
    std::string sep;
    for (auto &attr : *output_attrs) {
      os << sep << ToString(attr);
      sep = "\n";
    }
    SHERPA_ONNX_LOGE("\n----------Model outputs info----------\n%s",
                     os.str().c_str());
  }
}

rknn_custom_string GetCustomString(rknn_context ctx, bool debug) {
  rknn_custom_string custom_string;
  auto ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string,
                        sizeof(custom_string));
  SHERPA_ONNX_RKNN_CHECK(ret, "Failed to read custom string from the model");
  if (debug) {
    SHERPA_ONNX_LOGE("customs string: %s", custom_string.string);
  }
  return custom_string;
}

void SetCoreMask(rknn_context ctx, int32_t num_threads) {
  int32_t ret = RKNN_SUCC;
  switch (num_threads) {
    case 1:
      ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO);
      break;
    case 0:
      ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0);
      break;
    case -1:
      ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_1);
      break;
    case -2:
      ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_2);
      break;
    case -3:
      ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1);
      break;
    case -4:
      ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
      break;
    default:
      SHERPA_ONNX_LOGE(
          "Valid num_threads for rk npu is 1 (auto), 0 (core 0), -1 (core "
          "1), -2 (core 2), -3 (core 0_1), -4 (core 0_1_2). Given: %d",
          num_threads);
      break;
  }
  if (ret != RKNN_SUCC) {
    SHERPA_ONNX_LOGE(
        "Failed to select npu core to run the model (You can ignore it if "
        "you are not using RK3588.");
  }
}

}  // namespace sherpa_onnx
