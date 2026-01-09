// sherpa-onnx/csrc/math.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-onnx/csrc/math.h"

#include <vector>

#include "Eigen/Dense"

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

void ComputeMeanAndInvStd(const float *p, int32_t num_rows, int32_t num_cols,
                          std::vector<float> *mean,
                          std::vector<float> *inv_stddev) {
  using RowMajorMat =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Eigen::Map<const RowMajorMat> X(p, num_rows, num_cols);

  Eigen::RowVectorXf mean_vec = X.colwise().mean();

  Eigen::RowVectorXf mean_sq = X.array().square().colwise().mean();

  Eigen::RowVectorXf var = mean_sq.array() - mean_vec.array().square();

  Eigen::RowVectorXf stddev = var.array().max(0.0f).sqrt();

  Eigen::RowVectorXf inv_std = (stddev.array() + 1e-5f).inverse();

  mean->assign(mean_vec.data(), mean_vec.data() + num_cols);

  inv_stddev->assign(inv_std.data(), inv_std.data() + num_cols);
}

void NormalizeWhisperFeatures(float *features, int32_t num_frames,
                              int32_t feat_dim) {
  // log_spec = torch.clamp(features, min=1e-10).log10()
  // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
  // mel = (log_spec + 4.0) / 4.0

  using Eigen::ArrayXXf;
  using Eigen::Map;

  Map<ArrayXXf, Eigen::RowMajor> feats(features, num_frames, feat_dim);

  feats = feats.max(1e-10f).log10();

  float max_v = feats.maxCoeff() - 8.0f;

  feats = feats.max(max_v);
  feats = (feats + 4.0f) / 4.0f;
}

int32_t MaxElementIndex(const float *v, int32_t n) {
  // Map raw pointer to an Eigen vector (no copy)
  Eigen::Map<const Eigen::VectorXf> vec(v, n);

  Eigen::Index maxIndex;
  vec.maxCoeff(&maxIndex);

  return static_cast<int32_t>(maxIndex);
}

}  // namespace sherpa_onnx
