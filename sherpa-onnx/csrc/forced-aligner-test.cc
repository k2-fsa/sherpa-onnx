// sherpa-onnx/csrc/forced-aligner-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/forced-aligner.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(ForcedAlignerTest, ToTextGridFormat) {
  ForcedAlignmentResult r;
  r.duration = 5.61f;

  AlignmentTier phones;
  phones.name = "phones";
  phones.intervals.push_back({0.0f, 0.31f, "sil"});
  phones.intervals.push_back({0.31f, 0.35f, "t"});

  AlignmentTier words;
  words.name = "words";
  words.intervals.push_back({0.31f, 0.35f, "对"});

  r.tiers = {phones, words};

  const std::string expected =
      "File type = \"ooTextFile\"\n"
      "Object class = \"TextGrid\"\n"
      "\n"
      "xmin = 0\n"
      "xmax = 5.6100\n"
      "tiers? <exists>\n"
      "size = 2\n"
      "item []:\n"
      "    item [1]:\n"
      "        class = \"IntervalTier\"\n"
      "        name = \"phones\"\n"
      "        xmin = 0\n"
      "        xmax = 5.6100\n"
      "        intervals: size = 2\n"
      "        intervals [1]:\n"
      "            xmin = 0.0000\n"
      "            xmax = 0.3100\n"
      "            text = \"sil\"\n"
      "        intervals [2]:\n"
      "            xmin = 0.3100\n"
      "            xmax = 0.3500\n"
      "            text = \"t\"\n"
      "    item [2]:\n"
      "        class = \"IntervalTier\"\n"
      "        name = \"words\"\n"
      "        xmin = 0\n"
      "        xmax = 5.6100\n"
      "        intervals: size = 1\n"
      "        intervals [1]:\n"
      "            xmin = 0.3100\n"
      "            xmax = 0.3500\n"
      "            text = \"对\"\n";

  EXPECT_EQ(r.ToTextGrid(), expected);
}

TEST(ForcedAlignerTest, ToTextGridEscapesQuotes) {
  ForcedAlignmentResult r;
  r.duration = 1.0f;

  AlignmentTier words;
  words.name = "words";
  words.intervals.push_back({0.0f, 1.0f, "a\"b"});
  r.tiers = {words};

  // A double quote inside a label is written as two double quotes.
  std::string textgrid = r.ToTextGrid();
  EXPECT_NE(textgrid.find("text = \"a\"\"b\""), std::string::npos);
}

TEST(ForcedAlignerTest, MissingFilesThrow) {
  {
    // empty model_dir
    ForcedAlignerConfig config;
    config.model_dir = "";
    config.lexicon = "lexicon.txt";
    EXPECT_THROW(ForcedAligner{config}, std::invalid_argument);
  }
  {
    // empty lexicon
    ForcedAlignerConfig config;
    config.model_dir = "/tmp";
    config.lexicon = "";
    EXPECT_THROW(ForcedAligner{config}, std::invalid_argument);
  }
  {
    // nonexistent model dir (missing meta.json)
    ForcedAlignerConfig config;
    config.model_dir = "/no/such/dir";
    config.lexicon = "/no/such/lexicon.txt";
    EXPECT_THROW(ForcedAligner{config}, std::runtime_error);
  }
}

}  // namespace sherpa_onnx
