// sherpa-onnx/csrc/stack.cc

#include "sherpa-onnx/csrc/stack.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <utility>

#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

// static bool Compare(const std::vector<int64_t> &a,
//                     const std::vector<int64_t> &b, int32_t skip_dim) {
//   if (a.size() != b.size()) return false;

//   for (int32_t i = 0; i != static_cast<int32_t>(a.size()); ++i) {
//     if (i == skip_dim) continue;

//     if (a[i] != b[i]) return false;
//   }

//   return true;
// }

static void PrintShape(const std::vector<int64_t> &a) {
  for (auto i : a) {
    fprintf(stderr, "%d ", static_cast<int32_t>(i));
  }
  fprintf(stderr, "\n");
}

template <typename T /*=float*/>
Ort::Value Stack(OrtAllocator *allocator,
                 const std::vector<const Ort::Value *> &values, int32_t dim) {
  // TODO: not sure how to deal with this case yet
  // if (values.size() == 1u) {
  //   return Clone(allocator, values[0]);
  // }

  std::vector<int64_t> v0_shape =
      values[0]->GetTensorTypeAndShapeInfo().GetShape();

  std::vector<int64_t> ans_shape;
  ans_shape.reserve(v0_shape.size() + 1);
  ans_shape.insert(ans_shape.end(), v0_shape.data(), v0_shape.data() + dim);
  ans_shape.push_back(values.size());
  ans_shape.insert(ans_shape.end(), v0_shape.data() + dim, v0_shape.data() + v0_shape.size());

  auto leading_size = static_cast<int32_t>(std::accumulate(
      v0_shape.begin(), v0_shape.begin() + dim, 1, std::multiplies<int64_t>()));

  auto trailing_size = static_cast<int32_t>(
      std::accumulate(v0_shape.begin() + dim, v0_shape.end(), 1,
                      std::multiplies<int64_t>()));

  std::cout << "leading: " << leading_size << ", trailing: " << trailing_size << std::endl;
  for (const auto s: ans_shape) {
    std::cout << "   " << s;
  }
  std::cout << std::endl;

  Ort::Value ans = Ort::Value::CreateTensor<T>(allocator, ans_shape.data(), ans_shape.size());
  T *dst = ans.GetTensorMutableData<T>();

  for (int32_t i = 0; i != leading_size; ++i) {
    for (int32_t n = 0; n != static_cast<int32_t>(values.size()); ++n) {
      const T *src = values[n]->GetTensorData<T>();
      src += i * trailing_size;

      std::copy(src, src + trailing_size, dst);
      dst += trailing_size;
    }
  }

  return ans;
}

template Ort::Value Stack<float>(OrtAllocator *allocator,
                                 const std::vector<const Ort::Value *> &values,
                                 int32_t dim);

template Ort::Value Stack<int64_t>(OrtAllocator *allocator,
                                   const std::vector<const Ort::Value *> &values,
                                   int32_t dim);

}  // namespace sherpa_onnx
