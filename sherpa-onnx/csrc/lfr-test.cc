// sherpa-onnx/csrc/lfr-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/lfr.h"

#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(Lfr, PadsFirstAndLastFrames) {
  std::vector<float> input(13);
  std::iota(input.begin(), input.end(), 0.0f);

  const std::vector<float> expected = {
      0, 0, 0, 0, 1, 2, 3,
      3, 4, 5, 6, 7, 8, 9,
      9, 10, 11, 12, 12, 12, 12,
  };

  EXPECT_EQ(ApplyLfr(input, /*input_dim=*/1, /*window_size=*/7,
                     /*window_shift=*/6),
            expected);
}

TEST(Lfr, UsesCeilingOutputFrameCount) {
  const std::vector<std::pair<int32_t, int32_t>> test_cases = {
      {1, 1}, {5, 1}, {6, 1}, {7, 2}, {12, 2}, {13, 3}, {20, 4},
  };

  for (const auto &[input_frames, expected_output_frames] : test_cases) {
    std::vector<float> input(input_frames);
    std::iota(input.begin(), input.end(), 0.0f);

    auto output = ApplyLfr(input, /*input_dim=*/1, /*window_size=*/7,
                           /*window_shift=*/6);
    EXPECT_EQ(output.size(), expected_output_frames * 7)
        << "input_frames=" << input_frames;
  }
}

TEST(Lfr, RepeatsASingleInputFrame) {
  const std::vector<float> expected(7, 42.0f);

  EXPECT_EQ(ApplyLfr({42.0f}, /*input_dim=*/1, /*window_size=*/7,
                     /*window_shift=*/6),
            expected);
}

TEST(Lfr, PreservesFeatureDimensions) {
  const std::vector<float> input = {0, 1, 2, 3};
  const std::vector<float> expected = {0, 1, 0, 1, 2, 3};

  EXPECT_EQ(ApplyLfr(input, /*input_dim=*/2, /*window_size=*/3,
                     /*window_shift=*/2),
            expected);
}

TEST(Lfr, PadsFixedShapeWithZeros) {
  std::vector<float> input(13);
  std::iota(input.begin(), input.end(), 0.0f);
  auto expected = ApplyLfr(input, /*input_dim=*/1, /*window_size=*/7,
                           /*window_shift=*/6);
  expected.resize(5 * 7, 0.0f);

  EXPECT_EQ(ApplyLfrForFixedShape(input, /*input_dim=*/1,
                                  /*window_size=*/7, /*window_shift=*/6,
                                  /*output_frames=*/5),
            expected);
}

TEST(Lfr, TruncatesFixedShape) {
  std::vector<float> input(13);
  std::iota(input.begin(), input.end(), 0.0f);
  auto expected = ApplyLfr(input, /*input_dim=*/1, /*window_size=*/7,
                           /*window_shift=*/6);
  expected.resize(2 * 7);

  EXPECT_EQ(ApplyLfrForFixedShape(input, /*input_dim=*/1,
                                  /*window_size=*/7, /*window_shift=*/6,
                                  /*output_frames=*/2),
            expected);
}

TEST(Lfr, KeepsEmptyFixedShapeInputEmpty) {
  EXPECT_TRUE(ApplyLfrForFixedShape({}, /*input_dim=*/80,
                                    /*window_size=*/7, /*window_shift=*/6,
                                    /*output_frames=*/500)
                  .empty());
}

TEST(Lfr, EmptyInput) {
  EXPECT_TRUE(ApplyLfr({}, /*input_dim=*/80, /*window_size=*/7,
                       /*window_shift=*/6)
                  .empty());
}

TEST(Lfr, RejectsInvalidArgumentsAtRuntime) {
  EXPECT_DEATH_IF_SUPPORTED(
      ApplyLfr({1.0f}, /*input_dim=*/0, /*window_size=*/7,
               /*window_shift=*/6),
      "ApplyLfr: input_dim must be positive");
  EXPECT_DEATH_IF_SUPPORTED(
      ApplyLfr({1.0f}, /*input_dim=*/1, /*window_size=*/0,
               /*window_shift=*/6),
      "ApplyLfr: window_size must be positive");
  EXPECT_DEATH_IF_SUPPORTED(
      ApplyLfr({1.0f}, /*input_dim=*/1, /*window_size=*/7,
               /*window_shift=*/0),
      "ApplyLfr: window_shift must be positive");
  EXPECT_DEATH_IF_SUPPORTED(
      ApplyLfr({1.0f, 2.0f, 3.0f}, /*input_dim=*/2, /*window_size=*/7,
               /*window_shift=*/6),
      "ApplyLfr: input size .* is not divisible by input_dim 2");
  EXPECT_DEATH_IF_SUPPORTED(
      ApplyLfrForFixedShape({1.0f}, /*input_dim=*/1, /*window_size=*/7,
                            /*window_shift=*/6, /*output_frames=*/0),
      "ApplyLfrForFixedShape: output_frames must be positive");
  EXPECT_DEATH_IF_SUPPORTED(
      ApplyLfr({1.0f, 2.0f}, /*input_dim=*/2,
               /*window_size=*/std::numeric_limits<int32_t>::max(),
               /*window_shift=*/1),
      "ApplyLfr: output dimension .* exceeds int32 capacity");
}

}  // namespace sherpa_onnx
