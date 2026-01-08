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

#include "Eigen/Dense"

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

template <typename T>
void LogSoftmax(T *in, int32_t w, int32_t h) {
  for (int32_t i = 0; i != h; ++i) {
    LogSoftmax(in, w);
    in += w;
  }
}

template <typename T>
void SubtractBlank(T *in, int32_t w, int32_t h, int32_t blank_idx,
                   float blank_penalty) {
  for (int32_t i = 0; i != h; ++i) {
    in[blank_idx] -= blank_penalty;
    in += w;
  }
}

template <class T>
std::vector<int32_t> TopkIndex(const T *vec, int32_t size, int32_t topk) {
  std::vector<int32_t> vec_index(size);
  std::iota(vec_index.begin(), vec_index.end(), 0);

  std::partial_sort(vec_index.begin(), vec_index.begin() + topk,
                    vec_index.end(), [vec](int32_t index_1, int32_t index_2) {
                      return vec[index_1] > vec[index_2];
                    });

  int32_t k_num = std::min<int32_t>(size, topk);
  return {vec_index.begin(), vec_index.begin() + k_num};
}

template <class T>
std::vector<int32_t> TopkIndex(const std::vector<std::vector<T>> &vec,
                               int32_t topk) {
  std::vector<T> flatten;
  flatten.reserve(vec.size() * vec[0].size());
  for (const auto &v : vec) {
    flatten.insert(flatten.end(), v.begin(), v.end());
  }

  return TopkIndex(flatten.data(), flatten.size(), topk);
}

// in_out[i] += src[i] * scale
void ScaleAdd(const float *src, float scale, int32_t n, float *in_out);

// out[i] = src[i] * scale
void Scale(const float *src, float scale, int32_t n, float *out);

// For Paraformer
std::vector<float> ComputeAcousticEmbedding(
    const std::vector<float> &encoder_out, const std::vector<float> &alphas,
    int32_t encoder_dim);

// Transpose a 2-D matrix in row-major
std::vector<float> Transpose(const float *input, int32_t rows, int32_t cols);

/* Compute mean and inverse stddev over rows.
 *
 * @param p  A pointer to a 2-d array of shape (num_rows, num_cols)
 * @param num_rows Number of rows
 * @param num_cols Number of columns
 * @param mean On return, it contains p.mean(axis=0). You don't need to
 *             pre-allocate space for it.
 * @param inv_stddev On return, it contains 1/p.std(axis=0) You don't need to
 *                   pre-allocate space for it.
 */
void ComputeMeanAndInvStd(const float *p, int32_t num_rows, int32_t num_cols,
                          std::vector<float> *mean,
                          std::vector<float> *inv_stddev);

void NormalizeWhisperFeatures(float *features, int32_t num_frames,
                              int32_t feat_dim);

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_MATH_H_
