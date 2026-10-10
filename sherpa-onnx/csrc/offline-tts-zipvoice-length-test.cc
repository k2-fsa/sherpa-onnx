// Copyright (c) 2026 LittleMouse
#include "sherpa-onnx/csrc/offline-tts-zipvoice-length.h"

#include <limits>

#include "gtest/gtest.h"
namespace sherpa_onnx {
TEST(ZipvoiceStaticLength, DurationAndReferenceBudget) {
  EXPECT_EQ(ZipvoiceStaticFeatureLength(41, 40, 422, 1, 384, 1024, 620), 834);
  EXPECT_EQ(ZipvoiceStaticFeatureLength(41, 58, 422, 1, 384, 1024, 620), 1019);
  // Reject instead of squeezing a longer sentence into the decoder bucket.
  EXPECT_EQ(ZipvoiceStaticFeatureLength(41, 59, 422, 1, 384, 1024, 620), 0);
  // A shorter reference leaves decoder space but cannot enlarge the vocoder.
  EXPECT_EQ(ZipvoiceStaticFeatureLength(10, 62, 100, 1, 384, 1024, 620), 720);
  EXPECT_EQ(ZipvoiceStaticFeatureLength(10, 63, 100, 1, 384, 1024, 620), 0);
  // Inverse STFT needs at least two generated frames.
  EXPECT_EQ(ZipvoiceStaticFeatureLength(10, 1, 10, 1, 384, 1024, 620), 0);
  EXPECT_EQ(ZipvoiceStaticFeatureLength(10, 2, 10, 1, 384, 1024, 620), 12);
}
TEST(ZipvoiceStaticLength, TokensAndInvalidInputs) {
  EXPECT_EQ(ZipvoiceStaticFeatureLength(200, 183, 200, 1, 384, 1024, 620), 383);
  EXPECT_EQ(ZipvoiceStaticFeatureLength(200, 184, 200, 1, 384, 1024, 620), 0);
  EXPECT_EQ(ZipvoiceStaticFeatureLength(0, 10, 100, 1, 384, 1024, 620), 0);
  EXPECT_EQ(ZipvoiceStaticFeatureLength(10, 0, 100, 1, 384, 1024, 620), 0);
  EXPECT_EQ(ZipvoiceStaticFeatureLength(10, 1, 1024, 1, 384, 1024, 620), 0);
  EXPECT_EQ(ZipvoiceStaticFeatureLength(10, 10, 100, 0, 384, 1024, 620), 0);
  EXPECT_EQ(
      ZipvoiceStaticFeatureLength(
          10, 10, 100, std::numeric_limits<float>::quiet_NaN(), 384, 1024, 620),
      0);
  EXPECT_EQ(ZipvoiceStaticFeatureLength(10, 10, 100, 3, 384, 1024, 620), 0);
}
}  // namespace sherpa_onnx
