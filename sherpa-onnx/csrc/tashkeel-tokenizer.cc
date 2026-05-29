// sherpa-onnx/csrc/tashkeel-tokenizer.cc
//
// Copyright (c)  2026  Matias Lin

#include "sherpa-onnx/csrc/tashkeel-tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

void BuildIndexMapFrom(const std::vector<std::string> &src_list,
                       std::unordered_map<std::string, uint64_t> &map) {
  const size_t src_list_size = src_list.size();
  map.reserve(src_list_size);
  for (size_t i = 0; i < src_list_size; ++i) {
    map.emplace(src_list[i], i);
  }
}

void BuildInverseMapOf(
    const std::unordered_map<std::string, std::string> &src_map,
    std::unordered_map<std::string, std::string> &trg_map) {
  trg_map.reserve(src_map.size());

  for (auto const &[k, v] : src_map) {
    trg_map.emplace(v, k);
  }
}

constexpr char32_t kAlefCp = 0x0627;
constexpr char32_t kAlefMaddaCp = 0x0622;
constexpr char32_t kAlefHamzaAboveCp = 0x0623;
constexpr char32_t kAlefHamzaBelowCp = 0x0625;
constexpr char32_t kLamCp = 0x0644;
constexpr char32_t kHamzaCp = 0x0621;
constexpr char32_t kGhainCp = 0x063A;
constexpr char32_t kFathataCp = 0x064B;
constexpr char32_t kTatweelCp = 0x0640;
constexpr char32_t kFehCp = 0x0641;
constexpr char32_t kSukunCp = 0x0652;
constexpr char32_t kDaggerAlefCp = 0x0670;
constexpr char32_t kWaslaCp = 0x0671;
constexpr char32_t kLamAlefMaddaAboveCp = 0xFEF5;
constexpr char32_t kLamAlefHamzaAboveCp = 0xFEF7;
constexpr char32_t kLamAlefHamzaBelowCp = 0xFEF9;
constexpr char32_t kLamAlefCp = 0xFEFB;

// Standard Buckwalter transliteration table
// Arabic Unicode codepoint to Buckwalter ASCII character.
// Source https://github.com/KentonMurray/Buckwalter/blob/master/buckwalter.py
const std::unordered_map<char32_t, char> kAr2BwTable = {
    {kHamzaCp, '\''},          // hamza
    {kAlefMaddaCp, '|'},       // alef madda
    {kAlefHamzaAboveCp, '>'},  // alef hamza above
    {0x0624, '&'},             // waw hamza
    {kAlefHamzaBelowCp, '<'},  // alef hamza below
    {0x0626, '}'},             // yeh hamza
    {kAlefCp, 'A'},            // alef
    {0x0628, 'b'},             // beh
    {0x0629, 'p'},             // teh marbuta
    {0x062A, 't'},             // teh
    {0x062B, 'v'},             // theh
    {0x062C, 'j'},             // jeem
    {0x062D, 'H'},             // hah
    {0x062E, 'x'},             // khah
    {0x062F, 'd'},             // dal
    {0x0630, '*'},             // thal
    {0x0631, 'r'},             // reh
    {0x0632, 'z'},             // zain
    {0x0633, 's'},             // seen
    {0x0634, '$'},             // sheen
    {0x0635, 'S'},             // sad
    {0x0636, 'D'},             // dad
    {0x0637, 'T'},             // tah
    {0x0638, 'Z'},             // zah
    {0x0639, 'E'},             // ain
    {kGhainCp, 'g'},           // ghain
    {kTatweelCp, '_'},         // tatweel
    {kFehCp, 'f'},             // feh
    {0x0642, 'q'},             // qaf
    {0x0643, 'k'},             // kaf
    {0x0644, 'l'},             // lam
    {0x0645, 'm'},             // meem
    {0x0646, 'n'},             // noon
    {0x0647, 'h'},             // heh
    {0x0648, 'w'},             // waw
    {0x0649, 'Y'},             // alef maksura
    {0x064A, 'y'},             // yeh
    {kFathataCp, 'F'},         // fathata
    {0x064C, 'N'},             // dammatan
    {0x064D, 'K'},             // kasratan
    {0x064E, 'a'},             // fatha
    {0x064F, 'u'},             // damma
    {0x0650, 'i'},             // kasra
    {0x0651, '~'},             // shadda
    {kSukunCp, 'o'},           // sukun
    {kDaggerAlefCp, '`'},      // superscript alef (dagger alef)
    {kWaslaCp, '{'},           // alef wasla
};

std::unordered_map<char, char32_t> BuildBw2ArTable() {
  std::unordered_map<char, char32_t> m;
  m.reserve(kAr2BwTable.size());
  for (const auto &[cp, c] : kAr2BwTable) {
    m.emplace(c, cp);
  }
  return m;
}

const std::unordered_map<char, char32_t> kBw2ArTable = BuildBw2ArTable();

bool IsSingleCharTashkeel(char c) {
  switch (c) {
    case 'F':
    case 'N':
    case 'K':
    case 'a':
    case 'u':
    case 'i':
    case '~':
    case 'o':
      return true;
    default:
      return false;
  }
}

inline constexpr std::string_view kNoTashkeelTag = "<NT>";
inline constexpr std::string_view kBOSTag = "<BOS>";
inline constexpr std::string_view kEOSTag = "<EOS>";
inline constexpr std::string_view kPADTag = "<PAD>";

}  // namespace

TashkeelTokenizer::TashkeelTokenizer() {
  letters_ = {std::string(kPADTag),
              std::string(kBOSTag),
              std::string(kEOSTag),
              " ",
              "$",
              "&",
              "'",
              "*",
              "<",
              ">",
              "A",
              "D",
              "E",
              "H",
              "S",
              "T",
              "Y",
              "Z",
              "b",
              "d",
              "f",
              "g",
              "h",
              "j",
              "k",
              "l",
              "m",
              "n",
              "p",
              "q",
              "r",
              "s",
              "t",
              "v",
              "w",
              "x",
              "y",
              "z",
              "|",
              "}",
              "<MASK>"};
  tashkeel_list_ = {std::string(kPADTag),
                    std::string(kBOSTag),
                    std::string(kEOSTag),
                    std::string(kNoTashkeelTag),
                    "<SD>",
                    "<SDD>",
                    "<SF>",
                    "<SFF>",
                    "<SK>",
                    "<SKK>",
                    "F",
                    "K",
                    "N",
                    "a",
                    "i",
                    "o",
                    "u",
                    "~"};
  shaddah_last_ = {"a~", "u~", "i~", "F~", "N~", "K~"};
  shaddah_first_ = {"~a", "~u", "~i", "~F", "~N", "~K"};
  tashkeel_chars_ = {"F", "N", "K", "a", "u", "i", "~", "o"};

  tags_ = {
      {"<SF>", "~a"},   // shaddah and fatha
      {"<SD>", "~u"},   // shaddah and Damma
      {"<SK>", "~i"},   // shaddah and kasra
      {"<SFF>", "~F"},  // shaddah and fathatayn
      {"<SDD>", "~N"},  // shaddah and Dammatayn
      {"<SKK>", "~K"}   // shaddah and kasratayn
  };

  BuildIndexMapFrom(letters_, letters_map_);
  BuildIndexMapFrom(tashkeel_list_, tashkeel_map_);
  BuildInverseMapOf(tags_, inverse_tags_);

  no_tashkeel_id_ = tashkeel_map_.at(std::string(kNoTashkeelTag));
  space_letter_id_ = letters_map_.at(" ");
}

std::string TashkeelTokenizer::Ar2Bw(const std::string &text) const {
  std::u32string codepoints = Utf8ToUtf32(text);
  std::string out;
  out.reserve(codepoints.size());

  for (char32_t cp : codepoints) {
    auto it = kAr2BwTable.find(cp);
    if (it != kAr2BwTable.end()) {
      out.push_back(it->second);
    } else if (cp < 0x80) {
      // ASCII (space, punctuation) no need to change anything
      out.push_back(static_cast<char>(cp));
    } else {
      // Ignore unknown Arabic codepoints, essentially filtering
      // anything outside the Arabic range. Same behavior as CATT.
    }
  }
  return out;
}

std::string TashkeelTokenizer::Bw2Ar(const std::string &text) const {
  std::u32string out_cp;
  out_cp.reserve(text.size());

  for (char c : text) {
    auto it = kBw2ArTable.find(c);
    if (it != kBw2ArTable.end()) {
      out_cp.push_back(it->second);
    } else {
      // Non-Buckwalter char (space, punctuation, BOS/EOS, etc.).
      // They are all ASCII so no need for mapping.
      out_cp.push_back(static_cast<char32_t>(static_cast<unsigned char>(c)));
    }
  }
  return Utf32ToUtf8(out_cp);
}

std::string TashkeelTokenizer::CleanText(const std::string &text) const {
  std::u32string codepoints = Utf8ToUtf32(text);
  std::u32string filtered;
  filtered.reserve(codepoints.size());
  for (char32_t cp : codepoints) {
    // strip tatweel
    if (cp == kTatweelCp) {
      continue;
    }

    // wasla maps to alef
    if (cp == kWaslaCp) {
      filtered.push_back(kAlefCp);
      continue;
    }

    // Decompose lam-alef ligatures into LAM + base alef variant so
    // Ar2Bw() can map them through the standard Buckwalter table.
    if (cp == kLamAlefMaddaAboveCp || cp == kLamAlefHamzaAboveCp ||
        cp == kLamAlefHamzaBelowCp || cp == kLamAlefCp) {
      filtered.push_back(kLamCp);
      switch (cp) {
        case kLamAlefMaddaAboveCp:
          filtered.push_back(kAlefMaddaCp);
          break;
        case kLamAlefHamzaAboveCp:
          filtered.push_back(kAlefHamzaAboveCp);
          break;
        case kLamAlefHamzaBelowCp:
          filtered.push_back(kAlefHamzaBelowCp);
          break;
        case kLamAlefCp:
          filtered.push_back(kAlefCp);
          break;
      }
      continue;
    }

    bool allowed = cp == U' ' || (cp >= kHamzaCp && cp <= kGhainCp) ||
                   (cp >= kFehCp && cp <= kSukunCp) || cp == kDaggerAlefCp;
    filtered.push_back(allowed ? cp : U' ');
  }

  // Collapse consecutive spaces and trim. Inserted spaces are ASCII spaces
  // to be able to collapse at byte level.
  std::string utf8 = Utf32ToUtf8(filtered);
  std::string result;
  result.reserve(utf8.size());
  bool prev_was_space = true;  // skip leading spaces
  for (char c : utf8) {
    if (c == ' ') {
      if (!prev_was_space) {
        result.push_back(' ');
        prev_was_space = true;
      }
    } else {
      result.push_back(c);
      prev_was_space = false;
    }
  }
  if (!result.empty() && result.back() == ' ') {
    result.pop_back();  // trim spacing
  }
  return result;
}

// Swaps the shaddah last with their respective shaddah first counterparts
std::string TashkeelTokenizer::UnifyShaddahPosition(
    const std::string &text) const {
  std::string out = text;
  for (size_t i = 0; i < shaddah_last_.size(); ++i) {
    const std::string &from = shaddah_last_[i];
    const std::string &to = shaddah_first_[i];
    size_t pos = 0;
    while ((pos = out.find(from, pos)) != std::string::npos) {
      out.replace(pos, from.size(), to);
      pos += to.size();
    }
  }
  return out;
}

std::string TashkeelTokenizer::DedupConsecutiveHarakat(
    const std::string &text) const {
  std::string out = text;
  for (const std::string &h : tashkeel_chars_) {
    std::string doubled = h + h;
    size_t pos = 0;
    while ((pos = out.find(doubled, pos)) != std::string::npos) {
      out.replace(pos, doubled.size(), h);
    }
  }
  return out;
}

TashkeelTokenizer::LetterTashkeelPairList
TashkeelTokenizer::SplitTashkeelFromText(const std::string &bw_text) const {
  LetterTashkeelPairList pairs;
  pairs.reserve(bw_text.size() + 2);
  pairs.emplace_back(std::string(kBOSTag), std::string(kBOSTag));

  const size_t bw_text_len = bw_text.size();
  for (size_t i = 0; i < bw_text_len; ++i) {
    char cur = bw_text[i];

    // Skip tashkeel chars
    if (IsSingleCharTashkeel(cur)) {
      continue;
    }

    // Peek ahead for an attached tashkeel
    char next = (i + 1 < bw_text_len) ? bw_text[i + 1] : '\0';
    if (next == '\0' || !IsSingleCharTashkeel(next)) {
      pairs.emplace_back(std::string(1, cur), kNoTashkeelTag);
      continue;
    }

    if (next == '~') {
      // Shaddah possibly followed by another harakat, we use a combined tag
      if (i + 2 < bw_text_len) {
        std::string key = std::string("~") + bw_text[i + 2];
        auto it = inverse_tags_.find(key);
        if (it != inverse_tags_.end()) {
          pairs.emplace_back(std::string(1, cur), it->second);
          continue;
        }
      }
      // Shaddah alone
      pairs.emplace_back(std::string(1, cur), "~");
    } else {
      // Regular harakat
      pairs.emplace_back(std::string(1, cur), std::string(1, next));
    }
  }
  pairs.emplace_back(std::string(kEOSTag), std::string(kEOSTag));
  return pairs;
}

std::string TashkeelTokenizer::CombineTashkeelWithText(
    const LetterTashkeelPairList &pairs) const {
  std::string out;
  out.reserve(pairs.size() * 2);
  for (const auto &[letter, tashkeel] : pairs) {
    out.append(letter);
    auto it = tags_.find(tashkeel);
    if (it != tags_.end()) {
      // Mapping multi-char (e.g. "<SF>" will map to ~a)
      out.append(it->second);
    } else if (tashkeel != kNoTashkeelTag) {
      // No need to map single-char harakat or stray special token
      out.append(tashkeel);
    }  // <NT> does not contribute
  }
  return out;
}

// Used by Decode to clean the output (e.g. sometimes the model might
// accidentally output "<BOS>" or "<EOS>" in non-edge positions).
std::vector<std::string> TashkeelTokenizer::FilterTashkeel(
    const std::vector<std::string> &tashkeel) const {
  std::vector<std::string> out;
  out.reserve(tashkeel.size());
  const size_t n_tashkeel = tashkeel.size();
  for (size_t i = 0; i < n_tashkeel; ++i) {
    std::string curr_t = tashkeel[i];
    if ((i != 0 && curr_t == std::string(kBOSTag)) ||
        (i != n_tashkeel - 1 && curr_t == std::string(kEOSTag))) {
      curr_t = kNoTashkeelTag;
    }
    out.push_back(std::move(curr_t));
  }
  return out;
}

TashkeelTokenizer::EncodeResult TashkeelTokenizer::Encode(
    const std::string &text) const {
  std::string cleaned = CleanText(text);
  std::string bw = Ar2Bw(cleaned);

  // Strip dagger alef (BW: '`')
  bw.erase(std::remove(bw.begin(), bw.end(), kAr2BwTable.at(kDaggerAlefCp)),
           bw.end());

  bw = UnifyShaddahPosition(bw);
  bw = DedupConsecutiveHarakat(bw);

  LetterTashkeelPairList pairs = SplitTashkeelFromText(bw);

  EncodeResult result;
  result.input_ids_.reserve(pairs.size());
  result.target_ids_.reserve(pairs.size());
  // Throws if the input was malformed (CleanText with Ar2Bw should guarantee
  // clean input)
  for (const auto &[letter, tashkeel] : pairs) {
    result.input_ids_.push_back(static_cast<int64_t>(letters_map_.at(letter)));
    result.target_ids_.push_back(
        static_cast<int64_t>(tashkeel_map_.at(tashkeel)));
  }
  return result;
}

std::string TashkeelTokenizer::Decode(
    const std::vector<int64_t> &input_ids,
    const std::vector<int64_t> &target_ids) const {
  // Map IDs back to letter and tashkeel strings
  std::vector<std::string> letters;
  letters.reserve(input_ids.size());
  for (int64_t id : input_ids) {
    letters.push_back(letters_.at(static_cast<size_t>(id)));
  }
  std::vector<std::string> tashkeels;
  tashkeels.reserve(target_ids.size());
  for (int64_t id : target_ids) {
    tashkeels.push_back(tashkeel_list_.at(static_cast<size_t>(id)));
  }

  // Maps any BOS/EOS interior (not at the ends) predictions in the tashkeels
  // to <NT>s. This is to preserve the alignment.
  tashkeels = FilterTashkeel(tashkeels);

  // Drop BOS/EOS/PAD from letters and tashkeels from the ends
  auto is_marker = [](const std::string &text) {
    return text == std::string(kBOSTag) || text == std::string(kEOSTag) ||
           text == std::string(kPADTag);
  };
  letters.erase(std::remove_if(letters.begin(), letters.end(), is_marker),
                letters.end());
  tashkeels.erase(std::remove_if(tashkeels.begin(), tashkeels.end(), is_marker),
                  tashkeels.end());

  // Combine letter and tashkeel pairs to be able to convert from BW to Arabic
  const size_t n_pairs = std::min(letters.size(), tashkeels.size());
  LetterTashkeelPairList pairs;
  pairs.reserve(n_pairs);
  for (size_t i = 0; i < n_pairs; ++i) {
    pairs.emplace_back(std::move(letters[i]), std::move(tashkeels[i]));
  }
  std::string bw_text = CombineTashkeelWithText(pairs);
  return Bw2Ar(bw_text);
}

int64_t TashkeelTokenizer::NoTashkeelId() const { return no_tashkeel_id_; }

int64_t TashkeelTokenizer::SpaceLetterId() const { return space_letter_id_; }

}  // namespace sherpa_onnx
