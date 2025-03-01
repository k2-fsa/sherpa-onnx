// sherpa-onnx/csrc/utils.cc
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/utils.h"

#include <sstream>
#include <unordered_map>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
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
    const rknn_custom_string &custom_string) {
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

  return ans;
}

}  // namespace sherpa_onnx
