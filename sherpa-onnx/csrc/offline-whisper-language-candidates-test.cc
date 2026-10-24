// sherpa-onnx/csrc/offline-whisper-language-candidates-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-language-candidates.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

const std::unordered_map<std::string, int32_t> kLang2ID = {
    {"en", 4}, {"ru", 1}, {"nl", 6}, {"de", 2}, {"yue", 7}};

TEST(ResolveWhisperLanguageCandidates, TrimsAndPreservesMetadataOrder) {
  std::vector<int32_t> ids;
  ASSERT_TRUE(ResolveWhisperLanguageCandidates(" nl,\tru , en\n,yue ", "", true,
                                             kLang2ID, &ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{6, 1, 4, 7}));
}

TEST(ResolveWhisperLanguageCandidates, AcceptsSingletonAndReplacesPriorList) {
  std::vector<int32_t> ids = {6, 1, 4};
  ASSERT_TRUE(ResolveWhisperLanguageCandidates(" de ", "", true, kLang2ID,
                                             &ids));
  EXPECT_EQ(ids, (std::vector<int32_t>{2}));
}

TEST(ResolveWhisperLanguageCandidates, RejectsEmptyCodesWithoutPartialList) {
  for (const char *csv : {"", " \t\n", ",en", "en,,ru", "en,", "en, ,ru"}) {
    SCOPED_TRACE(csv);
    std::vector<int32_t> ids = {6, 1, 4};
    EXPECT_FALSE(
        ResolveWhisperLanguageCandidates(csv, "", true, kLang2ID, &ids));
    EXPECT_TRUE(ids.empty());
  }
}

TEST(ResolveWhisperLanguageCandidates, RejectsDuplicateAfterTrim) {
  std::vector<int32_t> ids = {2};
  EXPECT_FALSE(ResolveWhisperLanguageCandidates("en,ru, en\t", "", true,
                                              kLang2ID, &ids));
  EXPECT_TRUE(ids.empty());
}

TEST(ResolveWhisperLanguageCandidates, RejectsUnknownCaseAndAliases) {
  for (const char *csv : {"en,xx", "EN", "en,Ru", "eng", "english"}) {
    SCOPED_TRACE(csv);
    std::vector<int32_t> ids = {6};
    EXPECT_FALSE(
        ResolveWhisperLanguageCandidates(csv, "", true, kLang2ID, &ids));
    EXPECT_TRUE(ids.empty());
  }
}

TEST(ResolveWhisperLanguageCandidates, RejectsCodesAbsentFromModelMetadata) {
  const std::unordered_map<std::string, int32_t> lang2id = {{"en", 4}};
  std::vector<int32_t> ids = {4};
  EXPECT_FALSE(
      ResolveWhisperLanguageCandidates("en,ru", "", true, lang2id, &ids));
  EXPECT_TRUE(ids.empty());
  EXPECT_FALSE(ResolveWhisperLanguageCandidates("en", "", true, {}, &ids));
  EXPECT_TRUE(ids.empty());
}

TEST(ResolveWhisperLanguageCandidates, RejectsFixedLanguage) {
  std::vector<int32_t> ids = {6};
  EXPECT_FALSE(ResolveWhisperLanguageCandidates("en,ru", "en", true, kLang2ID,
                                              &ids));
  EXPECT_TRUE(ids.empty());
}

TEST(ResolveWhisperLanguageCandidates, RejectsNonMultilingualModel) {
  std::vector<int32_t> ids = {6};
  EXPECT_FALSE(ResolveWhisperLanguageCandidates("en", "", false, kLang2ID,
                                              &ids));
  EXPECT_TRUE(ids.empty());
}

TEST(SelectWhisperLanguage, ChoosesBestCandidateDespiteHigherExcludedLogit) {
  const std::vector<float> logits = {100.0f, 3.0f, 9.0f, 0.0f,
                                     7.0f, 0.0f, 5.0f, 0.0f};
  std::vector<int32_t> ids;
  ASSERT_TRUE(ResolveWhisperLanguageCandidates("nl,ru,en", "", true, kLang2ID,
                                             &ids));
  EXPECT_EQ(SelectWhisperLanguage(logits.data(), ids), 4);
  ASSERT_TRUE(ResolveWhisperLanguageCandidates("nl,ru", "", true, kLang2ID,
                                             &ids));
  EXPECT_EQ(SelectWhisperLanguage(logits.data(), ids), 6);
  ASSERT_TRUE(ResolveWhisperLanguageCandidates("ru", "", true, kLang2ID, &ids));
  EXPECT_EQ(SelectWhisperLanguage(logits.data(), ids), 1);
}

TEST(SelectWhisperLanguage, ExactTiesFollowCandidateOrder) {
  const std::vector<float> logits = {100.0f, 7.0f, 0.0f, 0.0f,
                                     7.0f, 0.0f, 7.0f};
  std::vector<int32_t> ids;
  ASSERT_TRUE(ResolveWhisperLanguageCandidates("nl,ru,en", "", true, kLang2ID,
                                             &ids));
  EXPECT_EQ(SelectWhisperLanguage(logits.data(), ids), 6);
  ASSERT_TRUE(ResolveWhisperLanguageCandidates("en,ru,nl", "", true, kLang2ID,
                                             &ids));
  EXPECT_EQ(SelectWhisperLanguage(logits.data(), ids), 4);
}

}  // namespace
}  // namespace sherpa_onnx
