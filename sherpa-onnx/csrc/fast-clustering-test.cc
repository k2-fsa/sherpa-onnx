// sherpa-onnx/csrc/fast-clustering-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/fast-clustering.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(FastClustering, TestTwoClusters) {
  std::vector<float> features = {
      // point 0
      0.1,
      0.1,
      // point 2
      0.4,
      -0.5,
      // point 3
      0.6,
      -0.7,
      // point 1
      0.2,
      0.3,
  };

  FastClusteringConfig config;
  config.num_clusters = 2;

  FastClustering clustering(config);
  auto labels = clustering.Cluster(features.data(), 4, 2);
  int32_t k = 0;
  for (auto i : labels) {
    std::cout << "point " << k << ": label " << i << "\n";
    ++k;
  }
}

TEST(FastClustering, TestClusteringWithThreshold) {
  std::vector<float> features = {
      // point 0
      0.1,
      0.1,
      // point 2
      0.4,
      -0.5,
      // point 3
      0.6,
      -0.7,
      // point 1
      0.2,
      0.3,
  };

  FastClusteringConfig config;
  config.threshold = 0.5;

  FastClustering clustering(config);
  auto labels = clustering.Cluster(features.data(), 4, 2);
  int32_t k = 0;
  for (auto i : labels) {
    std::cout << "point " << k << ": label " << i << "\n";
    ++k;
  }
}

TEST(FastClustering, TestSilhouetteTwoClusters) {
  std::vector<float> features = {
      // cluster A
      1.0, 0.0,
      0.99, 0.01,
      0.98, -0.02,
      // cluster B
      -1.0, 0.0,
      -0.99, 0.01,
      -0.98, -0.02,
  };
  const int32_t num_rows = 6;
  const int32_t num_cols = 2;

  FastClusteringConfig config;
  config.num_clusters = 2;

  FastClustering clustering(config);
  std::vector<float> silhouettes;
  auto labels = clustering.Cluster(features.data(), num_rows, num_cols,
                                   &silhouettes);

  ASSERT_EQ(labels.size(), static_cast<size_t>(num_rows));
  ASSERT_EQ(silhouettes.size(), static_cast<size_t>(num_rows));

  double sum = 0;
  for (float sil : silhouettes) {
    EXPECT_GE(sil, -1.0f);
    EXPECT_LE(sil, 1.0f);
    EXPECT_FALSE(std::isnan(sil));
    sum += sil;
  }
  EXPECT_GT(sum / num_rows, 0.5);
}

TEST(FastClustering, TestSilhouetteSingletonClusters) {
  // Two singleton clusters. Silhouette is undefined for
  // singletons, so we return 0 for those rows.
  std::vector<float> features = {
      1.0, 0.0,
      -1.0, 0.0,
  };
  const int32_t num_rows = 2;
  const int32_t num_cols = 2;

  FastClusteringConfig config;
  config.num_clusters = 2;

  FastClustering clustering(config);
  std::vector<float> silhouettes;
  auto labels = clustering.Cluster(features.data(), num_rows, num_cols,
                                   &silhouettes);

  ASSERT_EQ(labels.size(), static_cast<size_t>(num_rows));
  ASSERT_EQ(silhouettes.size(), static_cast<size_t>(num_rows));
  EXPECT_FLOAT_EQ(silhouettes[0], 0.0f);
  EXPECT_FLOAT_EQ(silhouettes[1], 0.0f);
}

TEST(FastClustering, TestSilhouetteSingleCluster) {
  // When we have only one cluster, we cannot calculate the
  // silhouette coefficient. So, we expect the unavailable
  // sentinel.
  std::vector<float> features = {
      1.0, 0.0,
      0.99, 0.01,
      0.98, -0.02,
  };
  const int32_t num_rows = 3;
  const int32_t num_cols = 2;

  FastClusteringConfig config;
  config.num_clusters = 1;

  FastClustering clustering(config);
  std::vector<float> silhouettes;
  auto labels = clustering.Cluster(features.data(), num_rows, num_cols,
                                   &silhouettes);

  ASSERT_EQ(labels.size(), static_cast<size_t>(num_rows));
  ASSERT_EQ(silhouettes.size(), static_cast<size_t>(num_rows));
  for (float sil : silhouettes) {
    EXPECT_FLOAT_EQ(sil, kSilhouetteUnavailable);
  }
}
}  // namespace sherpa_onnx
