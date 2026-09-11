// sherpa-onnx/csrc/speaker-embedding-manager-test.cc
//
// Copyright (c) 2024 Jingzhao Ou (jingzhao.ou@gmail.com)

#include "sherpa-onnx/csrc/speaker-embedding-manager.h"

#include <cmath>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(SpeakerEmbeddingManager, AddAndRemove) {
  int32_t dim = 2;
  SpeakerEmbeddingManager manager(dim);
  std::vector<float> v = {0.1, 0.1};
  bool status = manager.Add("first", v.data());
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 1);

  // duplicate
  status = manager.Add("first", v.data());
  ASSERT_FALSE(status);
  ASSERT_EQ(manager.NumSpeakers(), 1);

  // non-duplicate
  v = {0.1, 0.9};
  status = manager.Add("second", v.data());
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 2);

  // do not exist
  status = manager.Remove("third");
  ASSERT_FALSE(status);

  status = manager.Remove("first");
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 1);

  v = {0.1, 0.1};
  status = manager.Add("first", v.data());
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 2);

  status = manager.Remove("first");
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 1);

  status = manager.Remove("second");
  ASSERT_TRUE(status);
  ASSERT_EQ(manager.NumSpeakers(), 0);
}

TEST(SpeakerEmbeddingManager, Search) {
  int32_t dim = 2;
  SpeakerEmbeddingManager manager(dim);
  std::vector<float> v1 = {0.1, 0.1};
  std::vector<float> v2 = {0.1, 0.9};
  std::vector<float> v3 = {0.9, 0.1};
  bool status = manager.Add("first", v1.data());
  ASSERT_TRUE(status);

  status = manager.Add("second", v2.data());
  ASSERT_TRUE(status);

  status = manager.Add("third", v3.data());
  ASSERT_TRUE(status);

  ASSERT_EQ(manager.NumSpeakers(), 3);

  std::vector<float> v = {15, 16};
  float threshold = 0.9;

  std::string name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "first");

  v = {2, 17};
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "second");

  v = {17, 2};
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "third");

  threshold = 0.9;
  v = {15, 16};
  status = manager.Remove("first");
  ASSERT_TRUE(status);
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "");

  v = {17, 2};
  status = manager.Remove("third");
  ASSERT_TRUE(status);
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "");

  v = {2, 17};
  status = manager.Remove("second");
  ASSERT_TRUE(status);
  name = manager.Search(v.data(), threshold);
  EXPECT_EQ(name, "");

  ASSERT_EQ(manager.NumSpeakers(), 0);
}

TEST(SpeakerEmbeddingManager, Verify) {
  int32_t dim = 2;
  SpeakerEmbeddingManager manager(dim);
  std::vector<float> v1 = {0.1, 0.1};
  std::vector<float> v2 = {0.1, 0.9};
  std::vector<float> v3 = {0.9, 0.1};
  bool status = manager.Add("first", v1.data());
  ASSERT_TRUE(status);

  status = manager.Add("second", v2.data());
  ASSERT_TRUE(status);

  status = manager.Add("third", v3.data());
  ASSERT_TRUE(status);

  std::vector<float> v = {15, 16};
  float threshold = 0.9;

  status = manager.Verify("first", v.data(), threshold);
  ASSERT_TRUE(status);

  v = {2, 17};
  status = manager.Verify("first", v.data(), threshold);
  ASSERT_FALSE(status);

  status = manager.Verify("second", v.data(), threshold);
  ASSERT_TRUE(status);

  v = {17, 2};
  status = manager.Verify("first", v.data(), threshold);
  ASSERT_FALSE(status);

  status = manager.Verify("second", v.data(), threshold);
  ASSERT_FALSE(status);

  status = manager.Verify("third", v.data(), threshold);
  ASSERT_TRUE(status);

  status = manager.Verify("fourth", v.data(), threshold);
  ASSERT_FALSE(status);
}

TEST(SpeakerEmbeddingManager, GetEmbedding) {
  int32_t dim = 2;
  SpeakerEmbeddingManager manager(dim);

  std::vector<float> out(dim, 0);
  ASSERT_FALSE(manager.GetEmbedding("missing", out.data()));

  std::vector<float> v1 = {0.1f, 0.1f};
  ASSERT_TRUE(manager.Add("first", v1.data()));

  ASSERT_TRUE(manager.GetEmbedding("first", out.data()));

  float norm = std::sqrt(out[0] * out[0] + out[1] * out[1]);
  EXPECT_NEAR(norm, 1.0f, 1e-5);

  // Cosine similarity with L2-normalized input should be ~1.
  float in_norm = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1]);
  float cosine =
      (out[0] * v1[0] + out[1] * v1[1]) / (norm * in_norm);
  EXPECT_NEAR(cosine, 1.0f, 1e-5);

  ASSERT_TRUE(manager.Remove("first"));
  ASSERT_FALSE(manager.GetEmbedding("first", out.data()));
}

TEST(SpeakerEmbeddingManager, GetEmbeddingFromList) {
  int32_t dim = 2;
  SpeakerEmbeddingManager manager(dim);

  // Average of (1,0) and (0,1) then L2-normalize → (1/√2, 1/√2)
  std::vector<std::vector<float>> list = {{1.0f, 0.0f}, {0.0f, 1.0f}};
  ASSERT_TRUE(manager.Add("spk", list));

  std::vector<float> out(dim, 0);
  ASSERT_TRUE(manager.GetEmbedding("spk", out.data()));

  float expected = 1.0f / std::sqrt(2.0f);
  EXPECT_NEAR(out[0], expected, 1e-5);
  EXPECT_NEAR(out[1], expected, 1e-5);
}

}  // namespace sherpa_onnx
