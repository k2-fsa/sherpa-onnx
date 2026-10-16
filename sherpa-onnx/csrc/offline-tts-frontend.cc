// sherpa-onnx/csrc/offline-tts-frontend.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-frontend.h"

#include <cstdlib>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-kitten-model-meta-data.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

std::string TokenIDs::ToString() const {
  std::ostringstream os;
  os << "TokenIDs(";
  os << "tokens=[";
  std::string sep;
  for (auto i : tokens) {
    os << sep << i;
    sep = ", ";
  }
  os << "], ";

  os << "tones=[";
  sep = {};
  for (auto i : tones) {
    os << sep << i;
    sep = ", ";
  }
  os << "]";
  os << ")";
  return os.str();
}

std::unordered_map<char32_t, int32_t> ReadPiperTokens(std::istream &is) {
  std::unordered_map<char32_t, int32_t> token2id;
  std::string line;
  std::string sym;
  std::u32string s;
  int32_t id = 0;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    iss >> sym;
    if (iss.eof()) {
      id = atoi(sym.c_str());
      sym = " ";
    } else {
      iss >> id;
    }

    iss >> std::ws;
    if (!iss.eof()) {
      SHERPA_ONNX_LOGE("Error when reading tokens: %s", line.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    s = Utf8ToUtf32(sym);
    if (s.size() != 1) {
      // for tokens.txt from coqui-ai/TTS, the last token is <BLNK>
      if (s.size() == 6 && s[0] == '<' && s[1] == 'B' && s[2] == 'L' &&
          s[3] == 'N' && s[4] == 'K' && s[5] == '>') {
        continue;
      }
      SHERPA_ONNX_LOGE("Error when reading tokens at Line %s. size: %d",
                       line.c_str(), static_cast<int32_t>(s.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    char32_t c = s[0];
    if (token2id.count(c)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(c));
      SHERPA_ONNX_EXIT(-1);
    }
    token2id.insert({c, id});
  }
  return token2id;
}

std::vector<int64_t> PiperPhonemesToIdsVits(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<char32_t> &phonemes, bool is_inflect) {
  // These markers must exist in tokens.txt; look them up defensively so a
  // vocab without them fails cleanly instead of throwing std::out_of_range
  // across the C ABI (which would call std::terminate).
  auto pad_it = token2id.find(U'_');
  if (pad_it == token2id.end()) {
    SHERPA_ONNX_LOGE("tokens.txt is missing the required '_' symbol.");
    return {};
  }
  int32_t pad = pad_it->second;
  int32_t bos = -1;
  int32_t eos = -1;
  if (!is_inflect) {
    auto bos_it = token2id.find(U'^');
    auto eos_it = token2id.find(U'$');
    if (bos_it == token2id.end() || eos_it == token2id.end()) {
      SHERPA_ONNX_LOGE("tokens.txt is missing the required '^' or '$' symbol.");
      return {};
    }
    bos = bos_it->second;
    eos = eos_it->second;
  }

  std::vector<int64_t> ans;
  ans.reserve(phonemes.size() * 2 + 3);

  if (is_inflect) {
    ans.push_back(pad);
  } else {
    ans.push_back(bos);
    ans.push_back(pad);
  }

  for (auto p : phonemes) {
    if (token2id.count(p)) {
      ans.push_back(token2id.at(p));
      ans.push_back(pad);
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }
  }

  if (!is_inflect) {
    ans.push_back(eos);
  }

  return ans;
}

std::vector<std::vector<int64_t>> PiperPhonemesToIdsMatcha(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<char32_t> &phonemes, bool use_eos_bos,
    int32_t max_token_len /*= 400*/) {
  std::vector<std::vector<int64_t>> ans;
  std::vector<int64_t> current;

  int32_t bos = -1;
  int32_t eos = -1;
  if (use_eos_bos && token2id.count(U'^') && token2id.count(U'$')) {
    bos = token2id.at(U'^');
    eos = token2id.at(U'$');
    current.push_back(bos);
  }

  for (auto p : phonemes) {
    if (token2id.count(p)) {
      current.push_back(token2id.at(p));
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }

    if (current.size() > max_token_len + 1) {
      if (eos >= 0) {
        current.push_back(eos);
      }
      ans.push_back(std::move(current));
      if (bos >= 0) {
        current.push_back(bos);
      }
    }
  }

  if (!current.empty()) {
    if (eos >= 0 && current.size() > 1) {
      current.push_back(eos);
      ans.push_back(std::move(current));
    } else {
      ans.push_back(std::move(current));
    }
  }

  return ans;
}

std::vector<std::vector<int64_t>> PiperPhonemesToIdsKitten(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<char32_t> &phonemes,
    const OfflineTtsKittenModelMetaData &meta_data) {
  std::vector<std::vector<int64_t>> ans;
  std::vector<int64_t> current;
  current.reserve(phonemes.size());
  current.push_back(meta_data.start_id);

  int32_t suffix_size = meta_data.add_pad_after_end ? 2 : 1;
  for (auto p : phonemes) {
    if (token2id.count(p)) {
      int32_t emitted_tokens = p == '.' ? 2 : 1;
      if (static_cast<int32_t>(current.size()) + emitted_tokens + suffix_size >
          meta_data.max_token_len) {
        current.push_back(meta_data.end_id);
        if (meta_data.add_pad_after_end) {
          current.push_back(meta_data.pad_id);
        }
        ans.push_back(std::move(current));
        current.reserve(phonemes.size());
        current.push_back(meta_data.start_id);
      }
      current.push_back(token2id.at(p));
      if (p == '.') {
        auto sp = token2id.find(' ');
        if (sp != token2id.end()) {
          current.push_back(sp->second);
        }
      }
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }
  }

  current.push_back(meta_data.end_id);
  if (meta_data.add_pad_after_end) {
    current.push_back(meta_data.pad_id);
  }
  ans.push_back(std::move(current));
  return ans;
}

std::vector<int64_t> CoquiPhonemesToIds(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<char32_t> &phonemes,
    const OfflineTtsVitsModelMetaData &vits_meta_data) {
  int32_t use_eos_bos = vits_meta_data.use_eos_bos;
  int32_t bos_id = vits_meta_data.bos_id;
  int32_t eos_id = vits_meta_data.eos_id;
  int32_t blank_id = vits_meta_data.blank_id;
  int32_t add_blank = vits_meta_data.add_blank;
  auto comma_it = token2id.find(',');
  if (comma_it == token2id.end()) {
    SHERPA_ONNX_LOGE("tokens.txt is missing the required ',' symbol.");
    return {};
  }
  int32_t comma_id = comma_it->second;

  std::vector<int64_t> ans;
  if (add_blank) {
    ans.reserve(phonemes.size() * 2 + 3);
  } else {
    ans.reserve(phonemes.size() + 2);
  }

  if (use_eos_bos) {
    ans.push_back(bos_id);
  }

  if (add_blank) {
    ans.push_back(blank_id);
    for (auto p : phonemes) {
      if (token2id.count(p)) {
        ans.push_back(token2id.at(p));
        ans.push_back(blank_id);
      } else {
        SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                         static_cast<uint32_t>(p));
      }
    }
  } else {
    for (auto p : phonemes) {
      if (token2id.count(p)) {
        ans.push_back(token2id.at(p));
      } else {
        SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                         static_cast<uint32_t>(p));
      }
    }
  }

  ans.push_back(comma_id);

  if (use_eos_bos) {
    ans.push_back(eos_id);
  }

  return ans;
}

}  // namespace sherpa_onnx
