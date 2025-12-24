// sherpa-onnx/csrc/math.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-onnx/csrc/math.h"

#include <vector>
namespace sherpa_onnx {

void ScaleAdd(const float *src, float scale, int32_t n, float *in_out) {
  Eigen::Map<const Eigen::ArrayXf> src_vec(src, n);
  Eigen::Map<Eigen::ArrayXf> inout_vec(in_out, n);

  inout_vec += scale * src_vec;
}

void Scale(const float *src, float scale, int32_t n, float *out) {
  Eigen::Map<const Eigen::ArrayXf> src_vec(src, n);
  Eigen::Map<Eigen::ArrayXf> out_vec(out, n);

  out_vec = scale * src_vec;
}

// this if for Paraformer
std::vector<float> ComputeAcousticEmbedding(
    const std::vector<float> &encoder_out, const std::vector<float> &alphas,
    int32_t encoder_dim) {
  std::vector<float> ans;
  ans.reserve(encoder_out.size());

  float acc = 0;
  std::vector<float> cur_emb(encoder_dim);
  for (int32_t i = 0; i < static_cast<int32_t>(alphas.size()); ++i) {
    float w = alphas[i];

    acc += w;
    if (acc >= 1) {
      float overflow = acc - 1;
      float remain = w - overflow;

      ScaleAdd(encoder_out.data() + i * encoder_dim, remain, encoder_dim,
               cur_emb.data());

      ans.insert(ans.end(), cur_emb.begin(), cur_emb.end());

      Scale(encoder_out.data() + i * encoder_dim, overflow, encoder_dim,
            cur_emb.data());

      acc = overflow;
    } else {
      ScaleAdd(encoder_out.data() + i * encoder_dim, w, encoder_dim,
               cur_emb.data());
    }
  }
  // TODO(fangjun): The last cur_emb is not used

  return ans;
}

std::vector<float> Transpose(const float *input, int32_t rows, int32_t cols) {
  // Written by ChatGPT
  std::vector<float> output(cols * rows);

  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      in(input, rows, cols);

  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      out(output.data(), cols, rows);

  out.noalias() = in.transpose();

  return output;
}

}  // namespace sherpa_onnx
