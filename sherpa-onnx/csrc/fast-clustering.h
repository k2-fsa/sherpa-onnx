// sherpa-onnx/csrc/fast-clustering.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FAST_CLUSTERING_H_
#define SHERPA_ONNX_CSRC_FAST_CLUSTERING_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/fast-clustering-config.h"

namespace sherpa_onnx {

// Sentinel used when the silhouette coefficient cannot be computed
// (e.g. only one cluster was formed). It is outside the valid silhouette
// range of [-1, 1].
inline constexpr float kSilhouetteUnavailable = -2.0f;

class FastClustering {
 public:
  explicit FastClustering(const FastClusteringConfig &config);
  ~FastClustering();

  /**
   * @param features Pointer to a 2-D feature matrix in row major. Each row
   *                 is a feature frame. It is changed in-place. We will
   *                 convert each feature frame to a normalized vector.
   *                 That is, the L2-norm of each vector will be equal to 1.
   *                 It uses cosine dissimilarity,
   *                 which is 1 - (cosine similarity)
   * @param num_rows Number of feature frames
   * @param num-cols The feature dimension.
   * @param silhouettes  Optional output. When non-null, on return it holds
   *                     num_rows silhouette coefficients, one per input row.
   *                     Valid values are [-1, 1]; singleton clusters and the
   *                     num_rows <= 1 case yield 0; the single-cluster case
   *                     yields kSilhouetteUnavailable.
   *
   * @return Return a vector of size num_rows. ans[i] contains the label
   *         for the i-th feature frame, i.e., the i-th row of the feature
   *         matrix.
   */
  std::vector<int32_t> Cluster(float *features, int32_t num_rows,
                               int32_t num_cols,
                               std::vector<float> *silhouettes = nullptr) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_FAST_CLUSTERING_H_
