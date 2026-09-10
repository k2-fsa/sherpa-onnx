// sherpa-onnx/csrc/fast-clustering.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/fast-clustering.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "Eigen/Dense"
#include "fastcluster-all-in-one.h"  // NOLINT

namespace sherpa_onnx {

class FastClustering::Impl {
 public:
  explicit Impl(const FastClusteringConfig &config) : config_(config) {}

  std::vector<int32_t> Cluster(float *features, int32_t num_rows,
                               int32_t num_cols,
                               std::vector<float> *silhouettes) const {
    if (silhouettes) {
      silhouettes->clear();
    }

    if (num_rows <= 0) {
      return {};
    }

    if (num_rows == 1) {
      if (silhouettes) {
        silhouettes->assign(1, 0.0f);
      }
      return {0};
    }

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m(features, num_rows, num_cols);
    m.rowwise().normalize();

    std::vector<double> distance((num_rows * (num_rows - 1)) / 2);

    int32_t k = 0;
    for (int32_t i = 0; i != num_rows; ++i) {
      auto v = m.row(i);
      for (int32_t j = i + 1; j != num_rows; ++j) {
        double cosine_similarity = v.dot(m.row(j));
        double consine_dissimilarity = 1 - cosine_similarity;

        if (consine_dissimilarity < 0) {
          consine_dissimilarity = 0;
        }

        distance[k] = consine_dissimilarity;
        ++k;
      }
    }

    std::vector<int32_t> merge(2 * (num_rows - 1));
    std::vector<double> height(num_rows - 1);

    fastclustercpp::hclust_fast(num_rows, distance.data(),
                                fastclustercpp::HCLUST_METHOD_COMPLETE,
                                merge.data(), height.data());

    std::vector<int32_t> labels(num_rows);
    if (config_.num_clusters > 0) {
      fastclustercpp::cutree_k(num_rows, merge.data(), config_.num_clusters,
                               labels.data());
    } else {
      fastclustercpp::cutree_cdist(num_rows, merge.data(), height.data(),
                                   config_.threshold, labels.data());
    }

    if (silhouettes) {
      ComputeSilhouettes(distance, labels, num_rows, silhouettes);
    }

    return labels;
  }

 private:
  // Compute per-point silhouette coefficients using the already computed
  // cosine-dissimilarity distance matrix.
  static void ComputeSilhouettes(const std::vector<double> &distance,
                                 const std::vector<int32_t> &labels,
                                 int32_t num_rows,
                                 std::vector<float> *silhouettes) {
    assert(silhouettes && "silhouettes output pointer must not be null");
    assert(!labels.empty() && "labels must not be empty");
    int32_t num_clusters = *std::max_element(labels.begin(), labels.end()) + 1;
    assert(num_clusters > 0 && "labels must contain non-negative cluster ids");

    const size_t total_elements = static_cast<size_t>(num_rows) * num_clusters;
    // Row-major: sum of distances from point i to points in cluster C
    std::vector<double> sum(total_elements, 0.0);
    // Row-major: number of points in cluster C contributing to the sum
    std::vector<int32_t> count(total_elements, 0);

    // We only need to calculate pairs row_index < col_index (no need for
    // self-pairs). The distance from point X to Y is the same distance from
    // point Y to X.
    int32_t distance_matrix_index = 0;
    for (int32_t row_index = 0; row_index != num_rows; ++row_index) {
      size_t row_offset_index = static_cast<size_t>(row_index) * num_clusters;
      for (int32_t col_index = row_index + 1; col_index != num_rows;
           ++col_index, ++distance_matrix_index) {
        double pair_distance = distance[distance_matrix_index];
        int32_t row_point_cluster = labels[row_index];
        int32_t col_point_cluster = labels[col_index];
        size_t col_offset_index = static_cast<size_t>(col_index) * num_clusters;

        sum[row_offset_index + col_point_cluster] += pair_distance;
        count[row_offset_index + col_point_cluster] += 1;

        sum[col_offset_index + row_point_cluster] += pair_distance;
        count[col_offset_index + row_point_cluster] += 1;
      }
    }

    silhouettes->assign(num_rows, 0.0f);
    for (int32_t row_index = 0; row_index != num_rows; ++row_index) {
      int32_t row_point_cluster = labels[row_index];
      size_t base = static_cast<size_t>(row_index) * num_clusters;

      int32_t own_count = count[base + row_point_cluster];
      if (own_count == 0) {
        // For singleton clusters, the coefficient is 0.0.
        continue;
      }

      double a = sum[base + row_point_cluster] / own_count;
      double b = std::numeric_limits<double>::infinity();

      for (int32_t cluster_index = 0; cluster_index != num_clusters;
           ++cluster_index) {
        if (cluster_index == row_point_cluster) continue;
        int32_t neighbor_count = count[base + cluster_index];
        if (neighbor_count == 0) continue;
        double mean = sum[base + cluster_index] / neighbor_count;
        if (mean < b) b = mean;
      }

      if (!std::isfinite(b)) {
        // We only have one cluster overall. The silhouette coefficient is
        // undefined, so we set it to kSilhouetteUnavailable (outside the
        // valid [-1, 1] range) as a the unavailable sentinel.
        (*silhouettes)[row_index] = kSilhouetteUnavailable;
        continue;
      }

      double denom = std::max(a, b);
      (*silhouettes)[row_index] =
          denom > 0 ? static_cast<float>((b - a) / denom) : 0.0f;
    }
  }

  FastClusteringConfig config_;
};

FastClustering::FastClustering(const FastClusteringConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FastClustering::~FastClustering() = default;

std::vector<int32_t> FastClustering::Cluster(
    float *features, int32_t num_rows, int32_t num_cols,
    std::vector<float> *silhouettes /*= nullptr*/) const {
  return impl_->Cluster(features, num_rows, num_cols, silhouettes);
}
}  // namespace sherpa_onnx
