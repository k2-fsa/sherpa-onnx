// sherpa-onnx/csrc/offline-whisper-timestamp-rules-test.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-timestamp-rules.h"

#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

// Realistic Whisper token IDs (from multilingual model)
constexpr int32_t kTimestampBegin = 50364;  // <|0.00|>
constexpr int32_t kEot = 50257;             // <|endoftranscript|>
constexpr int32_t kNoTimestamps = 50363;    // <|notimestamps|>
constexpr int32_t kVocabSize = 51865;
constexpr int32_t kSampleBegin = 3;  // After [sot, language, task]

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Helper to check if a logit is suppressed (is -inf)
bool IsSuppressed(float logit) {
  return std::isinf(logit) && logit < 0;
}

// Helper to count non-suppressed logits in a range
int32_t CountNonSuppressed(const float *logits, int32_t start, int32_t end) {
  int32_t count = 0;
  for (int32_t i = start; i < end; ++i) {
    if (!IsSuppressed(logits[i])) {
      ++count;
    }
  }
  return count;
}

// Helper to initialize logits with uniform values
void InitLogits(std::vector<float> *logits, float value = 0.0f) {
  logits->assign(kVocabSize, value);
}

class ApplyTimestampRulesTest : public ::testing::Test {
 protected:
  std::vector<float> logits_;

  void SetUp() override {
    InitLogits(&logits_);
  }
};

// =============================================================================
// Rule 1: Always suppress no_timestamps token
// =============================================================================

TEST_F(ApplyTimestampRulesTest, AlwaysSuppressNoTimestamps) {
  std::vector<int64_t> tokens = {1, 2, 3};  // SOT sequence only
  logits_[kNoTimestamps] = 5.0f;  // Give it a high value

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  EXPECT_TRUE(IsSuppressed(logits_[kNoTimestamps]));
}

// =============================================================================
// Rule 5: First sampled token must be a timestamp
// =============================================================================

TEST_F(ApplyTimestampRulesTest, FirstTokenMustBeTimestamp) {
  // Only SOT sequence, no sampled tokens yet
  std::vector<int64_t> tokens = {1, 2, 3};

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  // All text tokens should be suppressed
  for (int32_t i = 0; i < kTimestampBegin; ++i) {
    if (i != kNoTimestamps) {  // no_timestamps is already suppressed
      EXPECT_TRUE(IsSuppressed(logits_[i]))
          << "Text token " << i << " should be suppressed on first sample";
    }
  }

  // Timestamps within max_initial_timestamp_index should NOT be suppressed
  for (int32_t i = kTimestampBegin; i <= kTimestampBegin + 50; ++i) {
    EXPECT_FALSE(IsSuppressed(logits_[i]))
        << "Timestamp " << i << " should be allowed on first sample";
  }

  // Timestamps beyond max_initial_timestamp_index should be suppressed
  for (int32_t i = kTimestampBegin + 51; i < kVocabSize; ++i) {
    EXPECT_TRUE(IsSuppressed(logits_[i]))
        << "Timestamp " << i << " should be suppressed (beyond max_initial)";
  }
}

TEST_F(ApplyTimestampRulesTest, FirstTokenNoMaxInitialConstraint) {
  std::vector<int64_t> tokens = {1, 2, 3};

  // Pass -1 for max_initial_timestamp_index to disable the constraint
  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, -1);

  // All timestamps should be allowed
  for (int32_t i = kTimestampBegin; i < kVocabSize; ++i) {
    EXPECT_FALSE(IsSuppressed(logits_[i]))
        << "All timestamps should be allowed when max_initial is -1";
  }
}

// =============================================================================
// Rule 3: Timestamp pairing - after opening timestamp, force text
// =============================================================================

TEST_F(ApplyTimestampRulesTest, AfterFirstTimestampForceText) {
  // SOT sequence + first timestamp <|0.00|>
  std::vector<int64_t> tokens = {1, 2, 3, kTimestampBegin};

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  // All timestamps should be suppressed (force text)
  for (int32_t i = kTimestampBegin; i < kVocabSize; ++i) {
    EXPECT_TRUE(IsSuppressed(logits_[i]))
        << "Timestamp " << i << " should be suppressed after opening timestamp";
  }

  // Text tokens should NOT be suppressed (except no_timestamps)
  // Note: EOT is also a "text" token in this context
  int32_t text_allowed = CountNonSuppressed(logits_.data(), 0, kTimestampBegin);
  EXPECT_GT(text_allowed, 0) << "Some text tokens should be allowed";
}

TEST_F(ApplyTimestampRulesTest, AfterTwoConsecutiveTimestampsForceText) {
  // Pattern: <|0.00|><|0.00|> - two consecutive timestamps
  std::vector<int64_t> tokens = {1, 2, 3, kTimestampBegin, kTimestampBegin};

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  // All timestamps should be suppressed (force text)
  for (int32_t i = kTimestampBegin; i < kVocabSize; ++i) {
    EXPECT_TRUE(IsSuppressed(logits_[i]))
        << "Timestamp " << i << " should be suppressed after double timestamp";
  }
}

// =============================================================================
// Rule 3: After text+timestamp, force timestamp/EOT (suppress text)
// =============================================================================

TEST_F(ApplyTimestampRulesTest, AfterTextThenTimestampForceTimestampOrEot) {
  // Pattern: <|0.00|> "hello" <|2.00|> - segment just closed
  int32_t ts_0_00 = kTimestampBegin;
  int32_t ts_2_00 = kTimestampBegin + 100;  // 2.00 seconds = 100 * 0.02
  int32_t text_token = 500;  // some text token

  std::vector<int64_t> tokens = {1, 2, 3, ts_0_00, text_token, ts_2_00};

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  // Text tokens before EOT should be suppressed
  for (int32_t i = 0; i < kEot; ++i) {
    EXPECT_TRUE(IsSuppressed(logits_[i]))
        << "Text token " << i << " should be suppressed after segment closed";
  }

  // EOT should be allowed
  EXPECT_FALSE(IsSuppressed(logits_[kEot])) << "EOT should be allowed";

  // Text tokens after EOT but before timestamp_begin should be suppressed
  for (int32_t i = kEot + 1; i < kTimestampBegin; ++i) {
    EXPECT_TRUE(IsSuppressed(logits_[i]))
        << "Token " << i << " should be suppressed after segment closed";
  }

  // Timestamps >= last_ts should be allowed (monotonicity allows same ts)
  EXPECT_FALSE(IsSuppressed(logits_[ts_2_00]))
      << "Same timestamp should be allowed for next segment opening";
}

// =============================================================================
// Rule 4: Monotonicity - timestamps must not decrease
// =============================================================================

TEST_F(ApplyTimestampRulesTest, MonotonicityPreventsEarlierTimestamps) {
  // After <|0.00|> "text" - we're in text, last timestamp was 0.00
  int32_t ts_0_00 = kTimestampBegin;
  int32_t text_token = 500;

  std::vector<int64_t> tokens = {1, 2, 3, ts_0_00, text_token};

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  // Timestamps before ts_0_00 + 1 should be suppressed (strictly increasing)
  // Since last token was text, we require strictly increasing
  EXPECT_TRUE(IsSuppressed(logits_[ts_0_00]))
      << "Same timestamp should be suppressed when not closing segment";

  // Timestamps after should be allowed
  EXPECT_FALSE(IsSuppressed(logits_[ts_0_00 + 1]))
      << "Next timestamp should be allowed";
}

TEST_F(ApplyTimestampRulesTest, MonotonicityAllowsSameTimestampAfterClose) {
  // After <|0.00|> "text" <|2.00|> - segment just closed
  int32_t ts_0_00 = kTimestampBegin;
  int32_t ts_2_00 = kTimestampBegin + 100;
  int32_t text_token = 500;

  std::vector<int64_t> tokens = {1, 2, 3, ts_0_00, text_token, ts_2_00};

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  // Same timestamp should be allowed (for opening next segment)
  EXPECT_FALSE(IsSuppressed(logits_[ts_2_00]))
      << "Same timestamp allowed when segment just closed";

  // Earlier timestamps should still be suppressed
  EXPECT_TRUE(IsSuppressed(logits_[ts_2_00 - 1]))
      << "Earlier timestamps should be suppressed";
}

// =============================================================================
// Rule 6: Probability rule - force timestamp when sum > max text
// =============================================================================

TEST_F(ApplyTimestampRulesTest, ProbabilityRuleForcesTimestamp) {
  // Set up: we're in text (last token was not timestamp)
  int32_t ts_0_00 = kTimestampBegin;
  int32_t text_token = 500;

  std::vector<int64_t> tokens = {1, 2, 3, ts_0_00, text_token};

  // Give timestamps high logits, text tokens low logits
  for (int32_t i = 0; i < kTimestampBegin; ++i) {
    logits_[i] = -10.0f;
  }
  for (int32_t i = kTimestampBegin; i < kVocabSize; ++i) {
    logits_[i] = 0.0f;  // After logsumexp, this will dominate
  }

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  // Text tokens should be suppressed due to probability rule
  for (int32_t i = 0; i < kTimestampBegin; ++i) {
    EXPECT_TRUE(IsSuppressed(logits_[i]))
        << "Text token " << i << " should be suppressed by probability rule";
  }
}

TEST_F(ApplyTimestampRulesTest, ProbabilityRuleDoesNotApplyWhenTextDominates) {
  // Set up: we're in text, but text logits are higher
  int32_t ts_0_00 = kTimestampBegin;
  int32_t text_token = 500;

  std::vector<int64_t> tokens = {1, 2, 3, ts_0_00, text_token};

  // Give text tokens high logits, timestamps low
  for (int32_t i = 0; i < kTimestampBegin; ++i) {
    logits_[i] = 0.0f;
  }
  for (int32_t i = kTimestampBegin; i < kVocabSize; ++i) {
    logits_[i] = -100.0f;
  }

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  // Text tokens should NOT all be suppressed
  int32_t text_allowed = CountNonSuppressed(logits_.data(), 0, kTimestampBegin);
  EXPECT_GT(text_allowed, 0)
      << "Text tokens should be allowed when they dominate";
}

TEST_F(ApplyTimestampRulesTest, ProbabilityRuleSkippedAfterTimestamp) {
  // After timestamp, probability rule doesn't apply
  std::vector<int64_t> tokens = {1, 2, 3, kTimestampBegin};

  // Even with high timestamp logits, the pairing rule takes precedence
  for (int32_t i = kTimestampBegin; i < kVocabSize; ++i) {
    logits_[i] = 100.0f;
  }

  ApplyTimestampRules(logits_.data(), kVocabSize, tokens, kSampleBegin,
                      kTimestampBegin, kNoTimestamps, kEot, 50);

  // Timestamps should be suppressed (pairing rule), not text
  for (int32_t i = kTimestampBegin; i < kVocabSize; ++i) {
    EXPECT_TRUE(IsSuppressed(logits_[i]));
  }
}

// =============================================================================
// ParseTimestampTokens tests
// =============================================================================

class ParseTimestampTokensTest : public ::testing::Test {};

TEST_F(ParseTimestampTokensTest, BasicSingleSegment) {
  // <|0.00|> "hello" <|2.00|> EOT
  int32_t ts_0_00 = kTimestampBegin;
  int32_t ts_2_00 = kTimestampBegin + 100;
  std::vector<int32_t> tokens = {ts_0_00, 100, 200, 300, ts_2_00, kEot};

  auto segments = ParseTimestampTokens(tokens, kTimestampBegin, kEot);

  ASSERT_EQ(segments.size(), 1);
  EXPECT_FLOAT_EQ(segments[0].start_time, 0.0f);
  EXPECT_FLOAT_EQ(segments[0].end_time, 2.0f);
  ASSERT_EQ(segments[0].token_ids.size(), 3);
  EXPECT_EQ(segments[0].token_ids[0], 100);
  EXPECT_EQ(segments[0].token_ids[1], 200);
  EXPECT_EQ(segments[0].token_ids[2], 300);
}

TEST_F(ParseTimestampTokensTest, MultipleSegments) {
  // <|0.00|> "hi" <|1.00|><|1.00|> "bye" <|2.00|> EOT
  int32_t ts_0_00 = kTimestampBegin;
  int32_t ts_1_00 = kTimestampBegin + 50;
  int32_t ts_2_00 = kTimestampBegin + 100;
  std::vector<int32_t> tokens = {ts_0_00, 100, ts_1_00, ts_1_00, 200, ts_2_00, kEot};

  auto segments = ParseTimestampTokens(tokens, kTimestampBegin, kEot);

  ASSERT_EQ(segments.size(), 2);

  EXPECT_FLOAT_EQ(segments[0].start_time, 0.0f);
  EXPECT_FLOAT_EQ(segments[0].end_time, 1.0f);
  ASSERT_EQ(segments[0].token_ids.size(), 1);
  EXPECT_EQ(segments[0].token_ids[0], 100);

  EXPECT_FLOAT_EQ(segments[1].start_time, 1.0f);
  EXPECT_FLOAT_EQ(segments[1].end_time, 2.0f);
  ASSERT_EQ(segments[1].token_ids.size(), 1);
  EXPECT_EQ(segments[1].token_ids[0], 200);
}

TEST_F(ParseTimestampTokensTest, EotClosesOpenSegment) {
  // <|0.00|> "hello" EOT (no closing timestamp)
  int32_t ts_0_00 = kTimestampBegin;
  std::vector<int32_t> tokens = {ts_0_00, 100, 200, kEot};

  auto segments = ParseTimestampTokens(tokens, kTimestampBegin, kEot);

  ASSERT_EQ(segments.size(), 1);
  EXPECT_FLOAT_EQ(segments[0].start_time, 0.0f);
  // EOT closes the segment without a closing timestamp, so end_time is sentinel
  EXPECT_FLOAT_EQ(segments[0].end_time, -1.0f);
  ASSERT_EQ(segments[0].token_ids.size(), 2);
  EXPECT_EQ(segments[0].token_ids[0], 100);
  EXPECT_EQ(segments[0].token_ids[1], 200);
}

TEST_F(ParseTimestampTokensTest, EmptySegmentSkipped) {
  // <|0.00|><|1.00|><|1.00|> "text" <|2.00|> EOT
  // The first "segment" between 0.00 and 1.00 has no text, should be skipped
  int32_t ts_0_00 = kTimestampBegin;
  int32_t ts_1_00 = kTimestampBegin + 50;
  int32_t ts_2_00 = kTimestampBegin + 100;
  std::vector<int32_t> tokens = {ts_0_00, ts_1_00, ts_1_00, 100, ts_2_00, kEot};

  auto segments = ParseTimestampTokens(tokens, kTimestampBegin, kEot);

  ASSERT_EQ(segments.size(), 1);
  EXPECT_FLOAT_EQ(segments[0].start_time, 1.0f);
  EXPECT_FLOAT_EQ(segments[0].end_time, 2.0f);
}

TEST_F(ParseTimestampTokensTest, IncompleteSegmentGetsSentinel) {
  // <|0.00|> "hello" (no closing timestamp, no EOT)
  int32_t ts_0_00 = kTimestampBegin;
  std::vector<int32_t> tokens = {ts_0_00, 100, 200};

  auto segments = ParseTimestampTokens(tokens, kTimestampBegin, kEot);

  ASSERT_EQ(segments.size(), 1);
  EXPECT_FLOAT_EQ(segments[0].start_time, 0.0f);
  EXPECT_FLOAT_EQ(segments[0].end_time, -1.0f);  // Sentinel for incomplete
  ASSERT_EQ(segments[0].token_ids.size(), 2);
}

TEST_F(ParseTimestampTokensTest, SentinelConsistencyBetweenEotAndIncomplete) {
  // Verify that both EOT-closed and incomplete segments use the same sentinel
  // This ensures consistent handling by downstream code

  // Case 1: EOT-closed segment (no closing timestamp before EOT)
  int32_t ts_1_00 = kTimestampBegin + 50;
  std::vector<int32_t> tokens_eot = {ts_1_00, 100, kEot};
  auto segments_eot = ParseTimestampTokens(tokens_eot, kTimestampBegin, kEot);

  // Case 2: Incomplete segment (tokens end without closing timestamp or EOT)
  std::vector<int32_t> tokens_incomplete = {ts_1_00, 100};
  auto segments_incomplete =
      ParseTimestampTokens(tokens_incomplete, kTimestampBegin, kEot);

  ASSERT_EQ(segments_eot.size(), 1);
  ASSERT_EQ(segments_incomplete.size(), 1);

  // Both should have the same start_time
  EXPECT_FLOAT_EQ(segments_eot[0].start_time, 1.0f);
  EXPECT_FLOAT_EQ(segments_incomplete[0].start_time, 1.0f);

  // Both should use the same sentinel value for end_time
  EXPECT_FLOAT_EQ(segments_eot[0].end_time, -1.0f);
  EXPECT_FLOAT_EQ(segments_incomplete[0].end_time, -1.0f);
  EXPECT_FLOAT_EQ(segments_eot[0].end_time, segments_incomplete[0].end_time)
      << "EOT-closed and incomplete segments must use the same sentinel";
}

TEST_F(ParseTimestampTokensTest, NoSegmentsFromEmptyInput) {
  std::vector<int32_t> tokens = {};

  auto segments = ParseTimestampTokens(tokens, kTimestampBegin, kEot);

  EXPECT_EQ(segments.size(), 0);
}

TEST_F(ParseTimestampTokensTest, OnlyEot) {
  std::vector<int32_t> tokens = {kEot};

  auto segments = ParseTimestampTokens(tokens, kTimestampBegin, kEot);

  EXPECT_EQ(segments.size(), 0);
}

TEST_F(ParseTimestampTokensTest, TextBeforeFirstTimestampIgnored) {
  // Text tokens before any timestamp should be ignored
  int32_t ts_1_00 = kTimestampBegin + 50;
  int32_t ts_2_00 = kTimestampBegin + 100;
  std::vector<int32_t> tokens = {100, 200, ts_1_00, 300, ts_2_00, kEot};

  auto segments = ParseTimestampTokens(tokens, kTimestampBegin, kEot);

  ASSERT_EQ(segments.size(), 1);
  EXPECT_FLOAT_EQ(segments[0].start_time, 1.0f);
  EXPECT_FLOAT_EQ(segments[0].end_time, 2.0f);
  ASSERT_EQ(segments[0].token_ids.size(), 1);
  EXPECT_EQ(segments[0].token_ids[0], 300);
}

}  // namespace sherpa_onnx
