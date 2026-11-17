// sherpa-onnx/csrc/fa-lexicon-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/fa-lexicon.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/file-utils.h"

namespace sherpa_onnx {

static std::string WriteTempFile(const std::string &name,
                                 const std::string &content) {
  std::string path = ::testing::TempDir() + name;
  std::ofstream os = OpenOutputFile(path);
  os << content;
  return path;
}

static FaPronunciation MakePron(
    const std::vector<std::vector<std::string>> &groups) {
  FaPronunciation pron;
  pron.groups = groups;
  for (const auto &g : groups) {
    pron.phones.insert(pron.phones.end(), g.begin(), g.end());
  }
  return pron;
}

TEST(FaLexiconTest, ParseMultiPronunciation) {
  std::string path =
      WriteTempFile("fa-lexicon-parse1.dict",
                    "<unk> 0.99 0.17 1.0 1.0 spn\n"
                    "行 x a ŋ\n"
                    "行 h a ŋ\n"
                    "行 x a ŋ\n"  // duplicate pronunciation, deduplicated
                    // " || " separates per-character phone groups
                    "研究 || i n ɛ n || tɕ i o u\n"
                    "儿 ɚ\n");
  FaLexicon lexicon = FaLexicon::Load(path);
  EXPECT_TRUE(lexicon.Contains("行"));
  EXPECT_TRUE(lexicon.Contains("儿"));
  EXPECT_FALSE(lexicon.Contains("hang"));

  const auto &prons = lexicon.Get("行");
  EXPECT_EQ(prons.size(), 2u);  // two distinct pronunciations
  EXPECT_EQ(prons[0].phones, (std::vector<std::string>{"x", "a", "ŋ"}));
  EXPECT_EQ(prons[1].phones, (std::vector<std::string>{"h", "a", "ŋ"}));

  // " || " splits the phones into per-character groups.
  const auto &yan = lexicon.Get("研究");
  ASSERT_EQ(yan.size(), 1u);
  EXPECT_EQ(yan[0].phones, (std::vector<std::string>{"i", "n", "ɛ", "n", "tɕ",
                                                     "i", "o", "u"}));
  ASSERT_EQ(yan[0].groups.size(), 2u);
  EXPECT_EQ(yan[0].groups[0], (std::vector<std::string>{"i", "n", "ɛ", "n"}));
  EXPECT_EQ(yan[0].groups[1], (std::vector<std::string>{"tɕ", "i", "o", "u"}));

  // probability columns are dropped
  const auto &unk = lexicon.Get("<unk>");
  EXPECT_EQ(unk[0].phones, (std::vector<std::string>{"spn"}));
}

TEST(FaLexiconTest, ParseGroupsAndDeduplication) {
  std::string path = WriteTempFile(
      "fa-lexicon-parse2.dict",
      "字 i\n"
      "字 || i ||\n"  // same phones, same single group -> deduplicated
      "词 || t s || ɻ\n"
      "词 t s ɻ\n"      // same phones, different grouping -> deduplicated
      "甲乙 a b\n"
      "甲乙 || a || b\n"  // grouped form wins when it comes second
      "句 || tɕ y ||\n");  // trailing empty group is dropped
  FaLexicon lexicon = FaLexicon::Load(path);

  const auto &zi = lexicon.Get("字");
  EXPECT_EQ(zi.size(), 1u);
  EXPECT_EQ(zi[0].groups.size(), 1u);

  // The grouped form is kept regardless of the order of the lines.
  const auto &ci = lexicon.Get("词");
  EXPECT_EQ(ci.size(), 1u);
  EXPECT_EQ(ci[0].phones, (std::vector<std::string>{"t", "s", "ɻ"}));
  EXPECT_EQ(ci[0].groups.size(), 2u);

  const auto &jiayi = lexicon.Get("甲乙");
  EXPECT_EQ(jiayi.size(), 1u);
  ASSERT_EQ(jiayi[0].groups.size(), 2u);
  EXPECT_EQ(jiayi[0].groups[0], (std::vector<std::string>{"a"}));
  EXPECT_EQ(jiayi[0].groups[1], (std::vector<std::string>{"b"}));

  const auto &ju = lexicon.Get("句");
  ASSERT_EQ(ju[0].groups.size(), 1u);
  EXPECT_EQ(ju[0].groups[0], (std::vector<std::string>{"tɕ", "y"}));
}

TEST(FaLexiconTest, ParseErrors) {
  EXPECT_THROW(FaLexicon::Load(::testing::TempDir() + "no-such-file.dict"),
               std::runtime_error);

  std::string path = WriteTempFile("fa-lexicon-empty.dict", "\n   \n");
  EXPECT_THROW(FaLexicon::Load(path), std::runtime_error);
}

TEST(FaLexiconTest, TokenizeHanziPerCharacter) {
  FaLexicon lexicon;
  lexicon.entries["一"] = {MakePron({{"i"}})};
  lexicon.entries["点儿"] = {MakePron({{"t", "j"}, {"ɐ", "ɻ"}})};

  // One token per character; whitespace and punctuation are dropped.
  auto words = TokenizeFaText("一点儿，那么一。", lexicon);
  EXPECT_EQ(words,
            (std::vector<std::string>{"一", "点", "儿", "那", "么", "一"}));
}

TEST(FaLexiconTest, TokenizeHanziMixedScript) {
  FaLexicon lexicon;
  lexicon.entries["一"] = {MakePron({{"i"}})};

  // Non-hanzi characters become their own tokens in hanzi mode.
  auto words = TokenizeFaText("A一B", lexicon);
  EXPECT_EQ(words, (std::vector<std::string>{"A", "一", "B"}));
}

TEST(FaLexiconTest, TokenizeWhitespace) {
  FaLexicon lexicon;
  lexicon.entries["hello"] = {MakePron({{"h", "ə", "l", "oʊ"}})};
  lexicon.entries["world"] = {MakePron({{"w", "ɝ", "l", "d"}})};
  lexicon.entries["don't"] = {MakePron({{"d", "oʊ", "n", "t"}})};

  // Punctuation is stripped from word edges and case is folded; punctuation
  // inside a word is kept.
  auto words = TokenizeFaText("Hello, world! don't", lexicon);
  EXPECT_EQ(words, (std::vector<std::string>{"hello", "world", "don't"}));
}

TEST(FaLexiconTest, ResolveContextReadings) {
  // 行 has several readings; in the context 行长 only "hang" is kept.
  FaLexicon lexicon;
  lexicon.entries["行"] = {
      MakePron({{"ɕ", "i", "ŋ"}}), MakePron({{"x", "a", "ŋ"}}),
      MakePron({{"h", "a", "ŋ"}}), MakePron({{"x", "ə", "ŋ"}})};
  lexicon.entries["长"] = {MakePron({{"ʈʂ", "ʰ", "a", "ŋ"}}),
                           MakePron({{"ʈʂ", "a", "ŋ"}})};
  lexicon.entries["行长"] = {MakePron({{"x", "a", "ŋ"}, {"ʈʂ", "a", "ŋ"}}),
                             MakePron({{"h", "a", "ŋ"}, {"ʈʂ", "a", "ŋ"}})};

  auto resolved = ResolveFaPronunciations({"行", "长"}, lexicon);
  ASSERT_EQ(resolved.size(), 2u);
  // Context readings from the " || " groups of 行长 (set order).
  EXPECT_EQ(resolved[0], (std::vector<std::vector<std::string>>{
                             {"h", "a", "ŋ"}, {"x", "a", "ŋ"}}));
  EXPECT_EQ(resolved[1],
            (std::vector<std::vector<std::string>>{{"ʈʂ", "a", "ŋ"}}));

  // Without a multi-character context all readings are kept.
  auto alone = ResolveFaPronunciations({"行"}, lexicon);
  ASSERT_EQ(alone.size(), 1u);
  EXPECT_EQ(alone[0].size(), 4u);
}

TEST(FaLexiconTest, ResolveUnionsOverlappingContexts) {
  // The middle character is covered by two contexts; the union of both
  // context readings is kept.
  FaLexicon lexicon;
  lexicon.entries["甲"] = {MakePron({{"a"}}), MakePron({{"b"}})};
  lexicon.entries["乙"] = {MakePron({{"c"}}), MakePron({{"d"}})};
  lexicon.entries["丙"] = {MakePron({{"e"}}), MakePron({{"f"}})};
  lexicon.entries["甲乙"] = {MakePron({{"a"}, {"c"}})};
  lexicon.entries["乙丙"] = {MakePron({{"d"}, {"f"}})};

  auto resolved = ResolveFaPronunciations({"甲", "乙", "丙"}, lexicon);
  ASSERT_EQ(resolved.size(), 3u);
  EXPECT_EQ(resolved[0], (std::vector<std::vector<std::string>>{{"a"}}));
  EXPECT_EQ(resolved[1], (std::vector<std::vector<std::string>>{{"c"}, {"d"}}));
  EXPECT_EQ(resolved[2], (std::vector<std::vector<std::string>>{{"f"}}));
}

TEST(FaLexiconTest, ResolveIgnoresContextsWithoutGroups) {
  // A multi-character key whose phones are not split per character cannot
  // act as a context; all readings are kept.
  FaLexicon lexicon;
  lexicon.entries["甲"] = {MakePron({{"a"}}), MakePron({{"b"}})};
  lexicon.entries["乙"] = {MakePron({{"c"}}), MakePron({{"d"}})};
  lexicon.entries["甲乙"] = {MakePron({{"a", "c"}})};

  auto resolved = ResolveFaPronunciations({"甲", "乙"}, lexicon);
  ASSERT_EQ(resolved.size(), 2u);
  EXPECT_EQ(resolved[0].size(), 2u);
  EXPECT_EQ(resolved[1].size(), 2u);
}

TEST(FaLexiconTest, ResolveContextWithoutStandaloneEntry) {
  // The phrase has per-character groups but its characters have no
  // standalone entries; the context groups become their pronunciations.
  FaLexicon lexicon;
  lexicon.entries["甲乙"] = {MakePron({{"a"}, {"b"}})};

  auto resolved = ResolveFaPronunciations({"甲", "乙"}, lexicon);
  ASSERT_EQ(resolved.size(), 2u);
  EXPECT_EQ(resolved[0], (std::vector<std::vector<std::string>>{{"a"}}));
  EXPECT_EQ(resolved[1], (std::vector<std::vector<std::string>>{{"b"}}));
}

TEST(FaLexiconTest, ResolveOovYieldsEmpty) {
  FaLexicon lexicon;
  lexicon.entries["甲"] = {MakePron({{"a"}})};

  auto resolved = ResolveFaPronunciations({"甲", "乙"}, lexicon);
  ASSERT_EQ(resolved.size(), 2u);
  EXPECT_EQ(resolved[0].size(), 1u);
  EXPECT_TRUE(resolved[1].empty());
}

TEST(FaLexiconTest, BuildLexiconFstMultiPronunciation) {
  // word 1 has two pronunciations of different lengths; word 2 has one.
  std::vector<FaFstWord> words = {
      {1, {{2}, {3, 4}}},
      {2, {{5}}},
  };
  fst::VectorFst<fst::StdArc> fst = BuildFaLexiconFst(words, /*sil_id=*/1,
                                                      /*sil_prob=*/0.5f);

  // 3 base states (start/loop/sil) + 1 extra state for the 2-phone path.
  EXPECT_EQ(fst.NumStates(), 4);

  bool saw_word1 = false;
  bool saw_word2 = false;
  using Fst = fst::VectorFst<fst::StdArc>;
  for (fst::StateIterator<Fst> siter(fst); !siter.Done(); siter.Next()) {
    for (fst::ArcIterator<Fst> aiter(fst, siter.Value()); !aiter.Done();
         aiter.Next()) {
      const auto &arc = aiter.Value();
      if (arc.olabel == 1) {
        saw_word1 = true;
      }
      if (arc.olabel == 2) {
        saw_word2 = true;
      }
    }
  }
  EXPECT_TRUE(saw_word1);
  EXPECT_TRUE(saw_word2);

  // Illegal silence probabilities.
  EXPECT_THROW(BuildFaLexiconFst(words, 1, 1.0f), std::runtime_error);
  EXPECT_THROW(BuildFaLexiconFst(words, 1, 0.0f), std::runtime_error);
}

TEST(FaLexiconTest, BuildLexiconFstStructure) {
  // One single-phone word; check the silence structure and that the word
  // label sits on its only arc.
  std::vector<FaFstWord> words = {
      {7, {{3}}},
  };
  fst::VectorFst<fst::StdArc> fst = BuildFaLexiconFst(words, /*sil_id=*/9,
                                                      /*sil_prob=*/0.5f);
  using Fst = fst::VectorFst<fst::StdArc>;

  // start -> loop / sil; exactly one final state.
  int32_t num_final = 0;
  for (fst::StateIterator<Fst> siter(fst); !siter.Done(); siter.Next()) {
    if (fst.Final(siter.Value()) != fst::TropicalWeight::Zero()) {
      ++num_final;
    }
  }
  EXPECT_EQ(num_final, 1);

  bool saw_sil_arc = false;
  bool saw_word_arc = false;
  for (fst::StateIterator<Fst> siter(fst); !siter.Done(); siter.Next()) {
    for (fst::ArcIterator<Fst> aiter(fst, siter.Value()); !aiter.Done();
         aiter.Next()) {
      const auto &arc = aiter.Value();
      if (arc.ilabel == 9) {
        saw_sil_arc = true;
        EXPECT_EQ(arc.olabel, 0);
      }
      if (arc.olabel == 7) {
        saw_word_arc = true;
        EXPECT_EQ(arc.ilabel, 3);
      }
    }
  }
  EXPECT_TRUE(saw_sil_arc);
  EXPECT_TRUE(saw_word_arc);
}

}  // namespace sherpa_onnx
