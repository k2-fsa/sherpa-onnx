/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Daniel Povey)
 * Copyright (c)  2023                     (Pingfeng Luo)
 *
 */
// This file is copied from k2/csrc/utils.h
#ifndef SHERPA_ONNX_CSRC_MATH_H_
#define SHERPA_ONNX_CSRC_MATH_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

namespace sherpa_onnx {

// logf(FLT_EPSILON)
#define SHERPA_ONNX_MIN_LOG_DIFF_FLOAT -15.9423847198486328125f

// log(DBL_EPSILON)
#define SHERPA_ONNX_MIN_LOG_DIFF_DOUBLE \
  -36.0436533891171535515240975655615329742431640625

template <typename T>
struct LogAdd;

template <>
struct LogAdd<double> {
  double operator()(double x, double y) const {
    double diff;

    if (x < y) {
      diff = x - y;
      x = y;
    } else {
      diff = y - x;
    }
    // diff is negative.  x is now the larger one.

    if (diff >= SHERPA_ONNX_MIN_LOG_DIFF_DOUBLE) {
      double res;
      res = x + log1p(exp(diff));
      return res;
    }

    return x;  // return the larger one.
  }
};

template <>
struct LogAdd<float> {
  float operator()(float x, float y) const {
    float diff;

    if (x < y) {
      diff = x - y;
      x = y;
    } else {
      diff = y - x;
    }
    // diff is negative.  x is now the larger one.

    if (diff >= SHERPA_ONNX_MIN_LOG_DIFF_DOUBLE) {
      float res;
      res = x + log1pf(expf(diff));
      return res;
    }

    return x;  // return the larger one.
  }
};

template <class T>
void LogSoftmax(T *input, int32_t input_len) {
  assert(input);

  T m = *std::max_element(input, input + input_len);

  T sum = 0.0;
  for (int32_t i = 0; i < input_len; i++) {
    sum += exp(input[i] - m);
  }

  T offset = m + log(sum);
  for (int32_t i = 0; i < input_len; i++) {
    input[i] -= offset;
  }
}

template <class T>
std::vector<int32_t> TopkIndex(const T *vec, int32_t size, int32_t topk) {
  std::vector<int32_t> vec_index(size);
  std::iota(vec_index.begin(), vec_index.end(), 0);

  std::sort(vec_index.begin(), vec_index.end(),
            [vec](int32_t index_1, int32_t index_2) {
              return vec[index_1] > vec[index_2];
            });

  int32_t k_num = std::min<int32_t>(size, topk);
  std::vector<int32_t> index(vec_index.begin(), vec_index.begin() + k_num);
  return index;
}

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_MATH_H_
