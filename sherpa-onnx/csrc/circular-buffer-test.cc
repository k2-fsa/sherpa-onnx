// sherpa-onnx/csrc/circular-buffer-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/circular-buffer.h"

#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

TEST(CircularBuffer, Push) {
  CircularBuffer buffer(10);
  EXPECT_EQ(buffer.Size(), 0);
  EXPECT_EQ(buffer.Head(), 0);
  EXPECT_EQ(buffer.Tail(), 0);

  std::vector<float> a = {0, 1, 2, 3, 4, 5};
  buffer.Push(a.data(), a.size());

  EXPECT_EQ(buffer.Size(), 6);
  EXPECT_EQ(buffer.Head(), 0);
  EXPECT_EQ(buffer.Tail(), 6);

  auto c = buffer.Get(0, a.size());
  EXPECT_EQ(a.size(), c.size());
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a[i], c[i]);
  }

  std::vector<float> d = {-6, -7, -8, -9};
  buffer.Push(d.data(), d.size());

  c = buffer.Get(a.size(), d.size());
  EXPECT_EQ(d.size(), c.size());
  for (int32_t i = 0; i != d.size(); ++i) {
    EXPECT_EQ(d[i], c[i]);
  }
}

TEST(CircularBuffer, PushAndPop) {
  CircularBuffer buffer(5);
  std::vector<float> a = {0, 1, 2, 3};
  buffer.Push(a.data(), a.size());

  EXPECT_EQ(buffer.Size(), 4);
  EXPECT_EQ(buffer.Head(), 0);
  EXPECT_EQ(buffer.Tail(), 4);

  buffer.Pop(2);

  EXPECT_EQ(buffer.Size(), 2);
  EXPECT_EQ(buffer.Head(), 2);
  EXPECT_EQ(buffer.Tail(), 4);

  auto c = buffer.Get(2, 2);
  EXPECT_EQ(c.size(), 2);
  EXPECT_EQ(c[0], 2);
  EXPECT_EQ(c[1], 3);

  a = {10, 20, 30};
  buffer.Push(a.data(), a.size());
  EXPECT_EQ(buffer.Size(), 5);
  EXPECT_EQ(buffer.Head(), 2);
  EXPECT_EQ(buffer.Tail(), 7);

  c = buffer.Get(2, 5);
  EXPECT_EQ(c.size(), 5);
  EXPECT_EQ(c[0], 2);
  EXPECT_EQ(c[1], 3);
  EXPECT_EQ(c[2], 10);
  EXPECT_EQ(c[3], 20);
  EXPECT_EQ(c[4], 30);

  c = buffer.Get(3, 4);
  EXPECT_EQ(c.size(), 4);
  EXPECT_EQ(c[0], 3);
  EXPECT_EQ(c[1], 10);
  EXPECT_EQ(c[2], 20);
  EXPECT_EQ(c[3], 30);

  c = buffer.Get(4, 3);
  EXPECT_EQ(c.size(), 3);
  EXPECT_EQ(c[0], 10);
  EXPECT_EQ(c[1], 20);
  EXPECT_EQ(c[2], 30);

  buffer.Pop(4);
  EXPECT_EQ(buffer.Size(), 1);
  EXPECT_EQ(buffer.Head(), 6);
  EXPECT_EQ(buffer.Tail(), 7);

  c = buffer.Get(6, 1);
  EXPECT_EQ(c.size(), 1);
  EXPECT_EQ(c[0], 30);

  a = {100, 200, 300, 400};
  buffer.Push(a.data(), a.size());
  EXPECT_EQ(buffer.Size(), 5);

  EXPECT_EQ(buffer.Size(), 5);
  EXPECT_EQ(buffer.Head(), 6);
  EXPECT_EQ(buffer.Tail(), 11);

  c = buffer.Get(6, 5);
  EXPECT_EQ(c.size(), 5);
  EXPECT_EQ(c[0], 30);
  EXPECT_EQ(c[1], 100);
  EXPECT_EQ(c[2], 200);
  EXPECT_EQ(c[3], 300);
  EXPECT_EQ(c[4], 400);

  buffer.Pop(3);
  EXPECT_EQ(buffer.Size(), 2);
  EXPECT_EQ(buffer.Head(), 9);
  EXPECT_EQ(buffer.Tail(), 11);

  c = buffer.Get(10, 1);
  EXPECT_EQ(c.size(), 1);
  EXPECT_EQ(c[0], 400);

  a = {1000, 2000, 3000};
  buffer.Push(a.data(), a.size());

  EXPECT_EQ(buffer.Size(), 5);
  EXPECT_EQ(buffer.Head(), 9);
  EXPECT_EQ(buffer.Tail(), 14);

  buffer.Pop(1);

  EXPECT_EQ(buffer.Size(), 4);
  EXPECT_EQ(buffer.Head(), 10);
  EXPECT_EQ(buffer.Tail(), 14);

  a = {4000};

  buffer.Push(a.data(), a.size());
  EXPECT_EQ(buffer.Size(), 5);
  EXPECT_EQ(buffer.Head(), 10);
  EXPECT_EQ(buffer.Tail(), 15);

  c = buffer.Get(13, 2);
  EXPECT_EQ(c.size(), 2);
  EXPECT_EQ(c[0], 3000);
  EXPECT_EQ(c[1], 4000);
}

// Regression test for
// https://github.com/k2-fsa/sherpa-onnx/issues/3635
//
// head_/tail_ used to be int32_t and overflowed to negative after 2^31
// samples (~37h at 16 kHz). tail_ % capacity then became negative and
// std::copy() wrote to an invalid iterator. Verify the counters survive the
// old 32-bit boundary.
TEST(CircularBuffer, PushAndGetAcrossInt32Boundary) {
  // Start just below the old overflow horizon: 2^31 - 10.
  const int64_t initial_position = (1LL << 31) - 10;
  CircularBuffer buffer(16, initial_position);

  EXPECT_EQ(buffer.Size(), 0);
  EXPECT_EQ(buffer.Head(), initial_position);
  EXPECT_EQ(buffer.Tail(), initial_position);

  // 20 samples straddle 2^31: a[i] lives at position initial_position + i.
  std::vector<float> a(20);
  for (int32_t i = 0; i != 20; ++i) {
    a[i] = static_cast<float>(i);
  }
  buffer.Push(a.data(), a.size());

  EXPECT_EQ(buffer.Size(), 20);
  EXPECT_EQ(buffer.Head(), initial_position);
  EXPECT_EQ(buffer.Tail(), initial_position + 20);

  // Read a window that spans the old boundary.
  int64_t before_boundary = (1LL << 31) - 2;  // a[8]
  auto c = buffer.Get(before_boundary, 6);
  ASSERT_EQ(c.size(), 6u);
  for (int32_t i = 0; i != 6; ++i) {
    EXPECT_EQ(c[i], static_cast<float>(before_boundary - initial_position + i));
  }

  // Pop across the boundary and keep reading.
  buffer.Pop(15);
  EXPECT_EQ(buffer.Head(), initial_position + 15);

  c = buffer.Get(buffer.Head(), 5);
  ASSERT_EQ(c.size(), 5u);
  for (int32_t i = 0; i != 5; ++i) {
    EXPECT_EQ(c[i], static_cast<float>(15 + i));
  }
}

// Same idea at a much larger position to prove 64-bit head_/tail_.
TEST(CircularBuffer, PushAndGetAtLargePosition) {
  const int64_t initial_position = 1LL << 40;
  CircularBuffer buffer(8, initial_position);

  std::vector<float> a = {1, 2, 3, 4, 5};
  buffer.Push(a.data(), a.size());

  EXPECT_EQ(buffer.Head(), initial_position);
  EXPECT_EQ(buffer.Tail(), initial_position + 5);

  auto c = buffer.Get(initial_position + 2, 3);
  ASSERT_EQ(c.size(), 3u);
  EXPECT_EQ(c[0], 3);
  EXPECT_EQ(c[1], 4);
  EXPECT_EQ(c[2], 5);
}

// Ring wrap-around combined with positions across the old 32-bit boundary.
TEST(CircularBuffer, RingWrapAcrossInt32Boundary) {
  const int64_t initial_position = (1LL << 31) - 4;
  CircularBuffer buffer(8, initial_position);

  std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  buffer.Push(a.data(), 6);  // wraps around inside the ring
  buffer.Pop(3);
  buffer.Push(a.data() + 6, 4);

  auto c = buffer.Get(buffer.Head(), buffer.Size());
  ASSERT_EQ(c.size(), 7u);
  const float expected[7] = {4, 5, 6, 7, 8, 9, 10};
  for (int32_t i = 0; i != 7; ++i) {
    EXPECT_EQ(c[i], expected[i]);
  }
  EXPECT_EQ(buffer.Head(), initial_position + 3);
  EXPECT_EQ(buffer.Tail(), initial_position + 10);
}

// Get() must reject a start_index that has already been popped.
TEST(CircularBuffer, GetRejectsPoppedIndex) {
  const int64_t initial_position = (1LL << 31) - 4;
  CircularBuffer buffer(8, initial_position);

  std::vector<float> a = {1, 2, 3, 4, 5};
  buffer.Push(a.data(), a.size());
  buffer.Pop(2);

  EXPECT_TRUE(buffer.Get(initial_position, 1).empty());
}

// Resize() of a non-empty buffer must also work at large head_/tail_.
TEST(CircularBuffer, ResizeAtLargePosition) {
  const int64_t initial_position = (1LL << 31) - 4;
  CircularBuffer buffer(8, initial_position);

  std::vector<float> a = {1, 2, 3, 4, 5, 6};
  buffer.Push(a.data(), 6);  // 6 <= 8, no resize
  std::vector<float> b = {7, 8, 9, 10, 11, 12};
  buffer.Push(b.data(), 6);  // 12 > 8, resizes 8 -> 16

  EXPECT_EQ(buffer.Size(), 12);
  EXPECT_EQ(buffer.Head(), initial_position);
  EXPECT_EQ(buffer.Tail(), initial_position + 12);

  auto c = buffer.Get(buffer.Head(), buffer.Size());
  ASSERT_EQ(c.size(), 12u);
  const float expected[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  for (int32_t i = 0; i != 12; ++i) {
    EXPECT_EQ(c[i], expected[i]) << "i = " << i;
  }
}

}  // namespace sherpa_onnx
