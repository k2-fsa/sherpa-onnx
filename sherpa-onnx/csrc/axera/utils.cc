// sherpa-onnx/csrc/axera/utils.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/utils.h"

#include <string.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/axera/io.hpp"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void ConvertNCHWtoNHWC(const float *src, int32_t n, int32_t channel,
                       int32_t height, int32_t width, float *dst) {
  for (int32_t i = 0; i < n; ++i) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t w = 0; w < width; ++w) {
        for (int32_t c = 0; c < channel; ++c) {
          dst[i * height * width * channel + h * width * channel + w * channel +
              c] = src[i * height * width * channel + c * height * width +
                       h * width + w];
        }
      }
    }
  }
}

std::string ToString(const AX_ENGINE_IO_INFO_T *io_info) {
  std::ostringstream os;
  os << "{";
  if (!io_info) {
    os << "null AX_ENGINE_IO_INFO_T}";
    return os.str();
  }

  os << "nInputSize: " << io_info->nInputSize;
  os << ", nOutputSize: " << io_info->nOutputSize;
  os << ", nMaxBatchSize: " << io_info->nMaxBatchSize;
  os << ", bDynamicBatchSize: "
     << (io_info->bDynamicBatchSize ? "true" : "false");
  os << "}";
  return os.str();
}

std::unordered_map<std::string, std::string> Parse(const char *custom_string,
                                                   bool debug /*= false*/) {
  std::unordered_map<std::string, std::string> ans;
  if (!custom_string) {
    SHERPA_ONNX_LOGE("Parse: custom_string is null");
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<std::string> fields;
  SplitStringToVector(custom_string, ";", false, &fields);
  std::vector<std::string> tmp;

  for (const auto &f : fields) {
    tmp.clear();
    SplitStringToVector(f, "=", false, &tmp);
    if (tmp.size() != 2) {
      SHERPA_ONNX_LOGE("Invalid custom string %s for %s", custom_string,
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

void InitEngine(bool debug) {
  AX_SYS_Init();
#ifdef AXERA_TARGET_CHIP_AX620E
  auto ret = AX_ENGINE_Init();
#else
  AX_ENGINE_NPU_ATTR_T npu_attr;
  memset(&npu_attr, 0, sizeof(npu_attr));
  npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
  auto ret = AX_ENGINE_Init(&npu_attr);
#endif
  if (ret != 0) {
    SHERPA_ONNX_LOGE("AX_ENGINE_Init failed, ret = %d", ret);
    SHERPA_ONNX_EXIT(-1);
  }
  if (debug) {
    SHERPA_ONNX_LOGE("AX_ENGINE_Init done.");
  }
}

void InitContext(void *model_data, size_t model_data_length, bool debug,
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
    SHERPA_ONNX_LOGE("AX_ENGINE_CreateHandle done. handle = %p",
                     (void *)(*handle));
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

  auto ret = AX_ENGINE_GetIOInfo(handle, io_info);
  if (ret != 0) {
    SHERPA_ONNX_LOGE("AX_ENGINE_GetIOInfo failed, ret = %d", ret);
    SHERPA_ONNX_EXIT(-1);
  }

  if (debug) {
    SHERPA_ONNX_LOGE("AX_ENGINE_GetIOInfo done.");
    SHERPA_ONNX_LOGE("IO_INFO: %s", ToString(*io_info).c_str());
    middleware::print_io_info(*io_info);
  }
}

void PrepareIO(AX_ENGINE_IO_INFO_T *io_info, AX_ENGINE_IO_T *io_data,
               bool debug) {
  if (!io_info || !io_data) {
    SHERPA_ONNX_LOGE("PrepareIO: io_info or io_data is null");
    SHERPA_ONNX_EXIT(-1);
  }

  auto ret = middleware::prepare_io(
      io_info, io_data,
      std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
  if (ret != 0) {
    SHERPA_ONNX_LOGE("middleware::prepare_io failed, ret = %d", ret);
    SHERPA_ONNX_EXIT(-1);
  }

  if (debug) {
    SHERPA_ONNX_LOGE("PrepareIO (middleware::prepare_io) done.");
  }
}

}  // namespace sherpa_onnx