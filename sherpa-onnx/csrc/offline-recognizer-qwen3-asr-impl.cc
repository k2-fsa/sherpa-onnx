// sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.cc
//
// Copyright (c)  2026 zengyw

#include "sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// Mel-frame chunk length (in frames) assumed by the Qwen3-ASR conv frontend
// when mapping log-mel features to audio tokens. Must match the chunk size
// baked into the exported ONNX graph; used by FeatToAudioTokensLen() to size
// the encoder mask.
constexpr int32_t kQwen3ChunkSize = 100;
// Number of mel bins per frame for Qwen3-ASR (Whisper-style log-mel). Must
// match the feature extractor (`WhisperTag` dim), `NormalizeWhisperFeatures`
// row width, and the last dimension of the conv-frontend ONNX input.
constexpr int32_t kQwen3MelDim = 128;

// Qwen3-ASR hotwords are placed in the system-role segment of the chat template
constexpr char kQwen3SystemPromptPrefix[] = "<|im_start|>system\n";
constexpr char kQwen3SystemPromptSuffix[] =
    "<|im_end|>\n<|im_start|>user\n<|audio_start|>";

// Format hotwords for the Qwen3 chat template: ASCII comma-separated list
// (e.g. "foo,bar,baz");
static std::string Qwen3FormatHotwordsForPrompt(const std::string &csv) {
  const std::vector<std::string> parts = SplitStringAndTrim(csv, ',');
  return Join(parts, " ");
}

static void Qwen3LogMaxTotalLenSuggestions(int32_t max_seq_len,
                                           int32_t model_max_len) {
  SHERPA_ONNX_LOGE(
      "The max_total_len (%d) caps prompt + audio KV (model limit %d). "
      "Suggestions:",
      max_seq_len, model_max_len);
  SHERPA_ONNX_LOGE(
      "  1) Reduce hotwords: fewer or shorter hotwords shorten the prompt.");
  SHERPA_ONNX_LOGE(
      "  2) Shorten audio: shorter clips yield fewer audio_token_len.");
  SHERPA_ONNX_LOGE(
      "  3) Re-export the Qwen3-ASR decoder ONNX with a larger max_total_len, "
      "raise --qwen3-asr-max-total-len (up to the model limit), and/or "
      "increase --qwen3-asr-max-new-tokens if generation is truncated.");
}

int32_t FeatToAudioTokensLen(int32_t feat_len, int32_t chunk_size) {
  if (feat_len <= 0 || chunk_size <= 0) {
    return 0;
  }

  auto conv_out_len_3x_stride2 = [](int32_t n) -> int32_t {
    int32_t x = (n + 1) / 2;
    x = (x + 1) / 2;
    return (x + 1) / 2;
  };

  auto aftercnn = [](int32_t x) -> int32_t {
    if (x <= 0) {
      return 0;
    }
    x = (x - 1) / 2 + 1;
    x = (x - 1) / 2 + 1;
    return (x - 1) / 2 + 1;
  };

  const int32_t cs = chunk_size;
  const int32_t full = feat_len / cs;
  const int32_t rem = feat_len % cs;
  const int32_t tn = conv_out_len_3x_stride2(cs);

  int32_t out = full * tn;
  if (rem > 0) {
    out += aftercnn(rem);
  }

  return std::max(out, 0);
}

inline bool IsFloatOrHalfBitsTensorType(ONNXTensorElementDataType elem_type) {
  return elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
}

inline float ReadFloatOrHalfBitsValue(const float *data_f32,
                                      const uint16_t *data_f16_bits,
                                      ONNXTensorElementDataType elem_type,
                                      int64_t index) {
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return data_f32[index];
  }

  return HalfBitsToFloat(data_f16_bits[index]);
}

}  // namespace

Ort::Value TrimAudioFeatures(Ort::Value audio_features, OrtAllocator *allocator,
                             bool *all_silent) {
  auto info = audio_features.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0) {
    return audio_features;
  }

  auto elem_type =
      static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (!IsFloatOrHalfBitsTensorType(elem_type)) {
    return audio_features;
  }

  const int32_t A = static_cast<int32_t>(shape[1]);
  const int32_t H = static_cast<int32_t>(shape[2]);

  const float *data_f32 = nullptr;
  const uint16_t *data_f16_bits = nullptr;
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    data_f32 = audio_features.GetTensorData<float>();
  } else {
    data_f16_bits = audio_features.GetTensorData<uint16_t>();
  }

  int32_t A_valid = 0;
  const float eps = 1e-6f;

  for (int32_t a = A - 1; a >= 0; --a) {
    float max_energy = 0.0f;
    for (int32_t h = 0; h < H; ++h) {
      float v = ReadFloatOrHalfBitsValue(data_f32, data_f16_bits, elem_type,
                                         static_cast<int64_t>(a) * H + h);
      float abs_val = std::abs(v);
      if (abs_val > max_energy) {
        max_energy = abs_val;
      }
    }

    if (max_energy > eps) {
      A_valid = a + 1;
      break;
    }
  }

  if (A_valid <= 0) {
    // The whole clip is silence. Report it so the caller can short-circuit
    // before building any hotwords/language prompt tokens, instead of
    // silently falling through to decoding on an all-silence input.
    if (all_silent != nullptr) {
      *all_silent = true;
    }
    return audio_features;
  }

  if (A_valid == A) {
    return audio_features;
  }

  std::array<int64_t, 3> new_shape{1, static_cast<int64_t>(A_valid), H};
  Ort::Value trimmed = Ort::Value::CreateTensor(allocator, new_shape.data(),
                                                new_shape.size(), elem_type);

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float *src = audio_features.GetTensorData<float>();
    float *dst = trimmed.GetTensorMutableData<float>();
    std::memcpy(
        dst, src,
        static_cast<size_t>(A_valid) * static_cast<size_t>(H) * sizeof(float));
  } else {
    const uint16_t *src = audio_features.GetTensorData<uint16_t>();
    uint16_t *dst = trimmed.GetTensorMutableData<uint16_t>();
    std::memcpy(dst, src,
                static_cast<size_t>(A_valid) * static_cast<size_t>(H) *
                    sizeof(uint16_t));
  }

  return trimmed;
}

namespace {

// Matches is_cjk_char() in qwen3_forced_aligner.py: CJK Unified Ideographs
// and extensions only. Deliberately narrower than IsCJK() in text-utils.h,
// which also covers Hangul syllables and kana -- the reference keeps those
// inside words.
bool IsAlignerCJKChar(char32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF);
}

// Whether |cp| is whitespace for the purpose of splitting aligner words.
// Covers ASCII whitespace plus the Unicode space separators that Python's
// str.split() treats as delimiters.
bool IsAlignerSpaceChar(char32_t cp) {
  return cp == U' ' || (cp >= 0x09 && cp <= 0x0D) || cp == 0x85 || cp == 0xA0 ||
         cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
         cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// Whether |cp| is kept in a forced-aligner word, mirroring is_kept_char()
// in qwen3_forced_aligner.py: "'" plus Unicode letter (L*) and number (N*)
// categories. Since C++ has no category table, this keeps everything except
// known punctuation, symbol, separator, control and format ranges -- which
// covers all scripts' letters and digits.
bool IsAlignerKeptChar(char32_t cp) {
  if (cp == U'\'') {
    return true;
  }

  if (cp < 0x80) {
    return std::isalnum(static_cast<unsigned char>(cp)) != 0;
  }

  if (IsAlignerSpaceChar(cp)) {
    return false;
  }

  // Other format/control characters
  if (cp == 0xAD || cp == 0x600 || cp == 0x601 || cp == 0x602 || cp == 0x603 ||
      cp == 0x604 || cp == 0x605 || cp == 0x61C || cp == 0x6DD || cp == 0x70F ||
      cp == 0x8E2 || cp == 0x180E || cp == 0x200B || cp == 0x200E ||
      cp == 0x200F || cp == 0xFEFF) {
    return false;
  }

  // Combining marks (M*): the reference drops them
  if ((cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x1AB0 && cp <= 0x1AFF) ||
      (cp >= 0x1DC0 && cp <= 0x1DFF) || (cp >= 0x20D0 && cp <= 0x20F0) ||
      (cp >= 0xFE20 && cp <= 0xFE2F)) {
    return false;
  }

  // Punctuation blocks
  if ((cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x2E00 && cp <= 0x2E7F) ||
      (cp >= 0x3000 && cp <= 0x303F) || (cp >= 0xFE10 && cp <= 0xFE1F) ||
      (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFE50 && cp <= 0xFE6F) ||
      (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
      (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65)) {
    return false;
  }

  // Symbol blocks (currency, arrows, math, technical, geometric, emoji, ...)
  if ((cp >= 0x20A0 && cp <= 0x20BF) || (cp >= 0x2100 && cp <= 0x214F) ||
      (cp >= 0x2190 && cp <= 0x2BFF) || (cp >= 0x1F000 && cp <= 0x1FAFF) ||
      (cp >= 0x1FB00 && cp <= 0x1FBFF) || cp == 0xA9 || cp == 0xAE ||
      cp == 0xB0 || cp == 0xB1 || cp == 0xB4 || cp == 0xB5 || cp == 0xB6 ||
      cp == 0xB7 || cp == 0xD7 || cp == 0xF7) {
    return false;
  }

  return true;
}

}  // namespace

std::vector<std::string> SplitQwen3AlignerWords(const std::string &text) {
  std::u32string u32 = Utf8ToUtf32(text);

  std::vector<std::string> words;
  std::u32string buf;

  // split_segment_with_chinese(): CJK ideographs become their own word;
  // consecutive non-CJK chars form one word.
  auto flush_buf = [&buf, &words]() {
    if (!buf.empty()) {
      words.push_back(Utf32ToUtf8(buf));
      buf.clear();
    }
  };

  size_t i = 0;
  while (i < u32.size()) {
    // tokenize_space_lang(): split on whitespace, clean each segment, then
    // split CJK characters out of it.
    while (i < u32.size() && IsAlignerSpaceChar(u32[i])) {
      ++i;
    }

    std::u32string seg;
    while (i < u32.size() && !IsAlignerSpaceChar(u32[i])) {
      if (IsAlignerKeptChar(u32[i])) {
        seg.push_back(u32[i]);
      }
      ++i;
    }

    for (char32_t cp : seg) {
      if (IsAlignerCJKChar(cp)) {
        flush_buf();
        words.push_back(Utf32ToUtf8(std::u32string(1, cp)));
      } else {
        buf.push_back(cp);
      }
    }
    flush_buf();
  }

  return words;
}

void FixQwen3AlignerTimestamps(std::vector<int64_t> *data) {
  const int32_t n = static_cast<int32_t>(data->size());
  if (n <= 1) {
    return;
  }

  // Longest non-decreasing subsequence (O(n^2); n = 2 * num_words is small).
  std::vector<int32_t> dp(n, 1);
  std::vector<int32_t> parent(n, -1);
  for (int32_t i = 1; i < n; ++i) {
    for (int32_t j = 0; j < i; ++j) {
      if ((*data)[j] <= (*data)[i] && dp[j] + 1 > dp[i]) {
        dp[i] = dp[j] + 1;
        parent[i] = j;
      }
    }
  }

  int32_t max_idx = 0;
  for (int32_t i = 1; i < n; ++i) {
    if (dp[i] > dp[max_idx]) {
      max_idx = i;
    }
  }

  std::vector<bool> is_normal(n, false);
  for (int32_t idx = max_idx; idx != -1; idx = parent[idx]) {
    is_normal[idx] = true;
  }

  std::vector<int64_t> &result = *data;
  int32_t i = 0;
  while (i < n) {
    if (is_normal[i]) {
      ++i;
      continue;
    }

    int32_t j = i;
    while (j < n && !is_normal[j]) {
      ++j;
    }
    const int32_t anomaly_count = j - i;

    bool has_left = false;
    int64_t left_val = 0;
    for (int32_t k = i - 1; k >= 0; --k) {
      if (is_normal[k]) {
        has_left = true;
        left_val = result[k];
        break;
      }
    }
    bool has_right = false;
    int64_t right_val = 0;
    for (int32_t k = j; k < n; ++k) {
      if (is_normal[k]) {
        has_right = true;
        right_val = result[k];
        break;
      }
    }

    if (anomaly_count <= 2) {
      for (int32_t k = i; k < j; ++k) {
        if (!has_left) {
          result[k] = right_val;
        } else if (!has_right) {
          result[k] = left_val;
        } else {
          result[k] = (k - (i - 1)) <= (j - k) ? left_val : right_val;
        }
      }
    } else {
      if (has_left && has_right) {
        const double step =
            static_cast<double>(right_val - left_val) / (anomaly_count + 1);
        for (int32_t k = i; k < j; ++k) {
          result[k] = static_cast<int64_t>(left_val + step * (k - i + 1));
        }
      } else if (has_left) {
        for (int32_t k = i; k < j; ++k) {
          result[k] = left_val;
        }
      } else if (has_right) {
        for (int32_t k = i; k < j; ++k) {
          result[k] = right_val;
        }
      }
    }

    i = j;
  }
}

namespace {

Ort::Value TruncateAudioFeatures(Ort::Value audio_features, int32_t keep_frames,
                                 OrtAllocator *allocator) {
  if (keep_frames <= 0) {
    return audio_features;
  }

  auto info = audio_features.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0) {
    return audio_features;
  }

  int32_t A = static_cast<int32_t>(shape[1]);
  int32_t H = static_cast<int32_t>(shape[2]);
  if (keep_frames >= A) {
    return audio_features;
  }

  auto elem_type =
      static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    return audio_features;
  }

  std::array<int64_t, 3> new_shape{1, static_cast<int64_t>(keep_frames), H};
  Ort::Value truncated = Ort::Value::CreateTensor(
      allocator, new_shape.data(), new_shape.size(),
      static_cast<ONNXTensorElementDataType>(elem_type));

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float *src = audio_features.GetTensorData<float>();
    float *dst = truncated.GetTensorMutableData<float>();
    std::memcpy(dst, src,
                static_cast<size_t>(keep_frames) * static_cast<size_t>(H) *
                    sizeof(float));
  } else {
    const uint16_t *src = audio_features.GetTensorData<uint16_t>();
    uint16_t *dst = truncated.GetTensorMutableData<uint16_t>();
    std::memcpy(dst, src,
                static_cast<size_t>(keep_frames) * static_cast<size_t>(H) *
                    sizeof(uint16_t));
  }

  return truncated;
}

Ort::Value BuildCachePosition(OrtAllocator *allocator, int32_t seq_len) {
  std::array<int64_t, 1> pos_shape{seq_len};
  Ort::Value cache_position = Ort::Value::CreateTensor<int64_t>(
      allocator, pos_shape.data(), pos_shape.size());

  int64_t *p = cache_position.GetTensorMutableData<int64_t>();
  std::iota(p, p + seq_len, int64_t{0});

  return cache_position;
}

inline float TensorAbsMax(const Ort::Value &t, int64_t limit) {
  auto info = t.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();

  int64_t n = 1;
  for (auto d : shape) {
    if (d <= 0) {
      return 0.0f;
    }
    if (n > (std::numeric_limits<int64_t>::max() / d)) {
      return 0.0f;
    }
    n *= d;
  }

  if (limit > 0 && n > limit) {
    n = limit;
  }

  auto elem_type =
      static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (!IsFloatOrHalfBitsTensorType(elem_type)) {
    return 0.0f;
  }

  const float *data_f32 = nullptr;
  const uint16_t *data_f16_bits = nullptr;
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    data_f32 = t.GetTensorData<float>();
  } else {
    data_f16_bits = t.GetTensorData<uint16_t>();
  }

  float abs_max = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    float v = std::abs(
        ReadFloatOrHalfBitsValue(data_f32, data_f16_bits, elem_type, i));
    if (std::isfinite(v) && v > abs_max) {
      abs_max = v;
    }
  }

  return abs_max;
}

// Number of trailing tokens to inspect for degenerate repetition, and the
// maximum number of distinct token ids such a window may contain before the
// decode is treated as a repetition loop rather than speech. A degenerate
// greedy decode cycles over two or three token ids until max_new_tokens;
// real transcriptions -- even of repetitive lyrics -- vary more than that
// within 64 consecutive tokens.
constexpr int32_t kQwen3LoopWindow = 64;
constexpr int32_t kQwen3LoopMaxDistinct = 3;

// Returns true if the last kQwen3LoopWindow entries of ids consist of at
// most kQwen3LoopMaxDistinct distinct token ids.
bool IsDegenerateRepetition(const std::vector<int64_t> &ids) {
  if (static_cast<int32_t>(ids.size()) < kQwen3LoopWindow) {
    return false;
  }

  std::array<int64_t, kQwen3LoopMaxDistinct> distinct{};
  int32_t num_distinct = 0;
  for (size_t i = ids.size() - kQwen3LoopWindow; i != ids.size(); ++i) {
    bool seen = false;
    for (int32_t k = 0; k != num_distinct; ++k) {
      if (distinct[k] == ids[i]) {
        seen = true;
        break;
      }
    }
    if (seen) {
      continue;
    }
    if (num_distinct == kQwen3LoopMaxDistinct) {
      return false;
    }
    distinct[num_distinct++] = ids[i];
  }
  return true;
}

// Removes the degenerate tail from ids: trailing tokens drawn from the token
// set of the final kQwen3LoopWindow entries, scanning no further back than
// that window so text decoded before the collapse is never removed even when
// it ends with a token the loop happens to reuse. What remains is the text
// decoded before the loop started (possibly nothing).
void TrimDegenerateTail(std::vector<int64_t> *ids) {
  std::array<int64_t, kQwen3LoopMaxDistinct> distinct{};
  int32_t num_distinct = 0;
  for (size_t i = ids->size() - kQwen3LoopWindow; i != ids->size(); ++i) {
    bool seen = false;
    for (int32_t k = 0; k != num_distinct; ++k) {
      if (distinct[k] == (*ids)[i]) {
        seen = true;
        break;
      }
    }
    if (!seen && num_distinct < kQwen3LoopMaxDistinct) {
      distinct[num_distinct++] = (*ids)[i];
    }
  }

  const size_t window_start = ids->size() - kQwen3LoopWindow;
  size_t keep = ids->size();
  while (keep > window_start) {
    bool in_loop_set = false;
    for (int32_t k = 0; k != num_distinct; ++k) {
      if (distinct[k] == (*ids)[keep - 1]) {
        in_loop_set = true;
        break;
      }
    }
    if (!in_loop_set) {
      break;
    }
    --keep;
  }
  ids->resize(keep);
}

inline void RemoveUtf8ReplacementChars(std::string *s) {
  if (!s || s->empty()) {
    return;
  }

  const std::string kReplacement = "\xEF\xBF\xBD";
  size_t pos = 0;
  while ((pos = s->find(kReplacement, pos)) != std::string::npos) {
    s->erase(pos, kReplacement.size());
  }
}

}  // namespace

OfflineRecognizerQwen3ASRImpl::OfflineRecognizerQwen3ASRImpl(
    const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(config),
      config_(config),
      model_(std::make_unique<OfflineQwen3ASRModel>(config.model_config)),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          config.model_config.qwen3_asr.tokenizer)),
      rng_(config.model_config.qwen3_asr.seed) {
  if (!config_.model_config.qwen3_asr.forced_aligner_decoder.empty()) {
    aligner_model_ =
        std::make_unique<OfflineQwen3ForcedAlignerModel>(config.model_config);
    aligner_tokenizer_ = std::make_unique<QwenAsrTokenizer>(
        config.model_config.qwen3_asr.forced_aligner_tokenizer);
  }
  InitPromptTemplateIds();
}

template <typename Manager>
OfflineRecognizerQwen3ASRImpl::OfflineRecognizerQwen3ASRImpl(
    Manager *mgr, const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(mgr, config),
      config_(config),
      model_(std::make_unique<OfflineQwen3ASRModel>(mgr, config.model_config)),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          mgr, config.model_config.qwen3_asr.tokenizer)),
      rng_(config.model_config.qwen3_asr.seed) {
  if (!config_.model_config.qwen3_asr.forced_aligner_decoder.empty()) {
    aligner_model_ = std::make_unique<OfflineQwen3ForcedAlignerModel>(
        mgr, config.model_config);
    aligner_tokenizer_ = std::make_unique<QwenAsrTokenizer>(
        mgr, config.model_config.qwen3_asr.forced_aligner_tokenizer);
  }
  InitPromptTemplateIds();
}

std::unique_ptr<OfflineStream> OfflineRecognizerQwen3ASRImpl::CreateStream()
    const {
  WhisperTag tag;
  tag.dim = kQwen3MelDim;
  // Qwen3-ASR's feature extractor uses a centered STFT; the kaldi-style
  // half-shift frame offset is enough to make the 1.7B model collapse into
  // repetition loops on some inputs (k2-fsa/sherpa-onnx#3535).
  tag.align_to_stft_center = true;
  return std::make_unique<OfflineStream>(tag);
}

void OfflineRecognizerQwen3ASRImpl::InitPromptTemplateIds() {
  const std::string audio_pad = "<|audio_pad|>";
  const std::string user_suffix = "<|audio_end|><|im_end|>\n";
  const std::string assistant_text = "<|im_start|>assistant\n";

  audio_pad_ids_ = tokenizer_->Encode(audio_pad);
  prompt_ids_after_ = tokenizer_->Encode(user_suffix + assistant_text);

  if (audio_pad_ids_.empty()) {
    SHERPA_ONNX_LOGE("Failed to tokenize <|audio_pad|> for qwen3-asr prompt");
    SHERPA_ONNX_EXIT(-1);
  }

  asr_text_token_id_ = tokenizer_->GetTokenId("<asr_text>");
  if (asr_text_token_id_ < 0) {
    SHERPA_ONNX_LOGE("Failed to locate <asr_text> token id for qwen3-asr");
    SHERPA_ONNX_EXIT(-1);
  }

  if (aligner_model_) {
    aligner_audio_start_token_id_ =
        aligner_tokenizer_->GetTokenId("<|audio_start|>");
    aligner_audio_end_token_id_ =
        aligner_tokenizer_->GetTokenId("<|audio_end|>");
    aligner_audio_pad_token_id_ =
        aligner_tokenizer_->GetTokenId("<|audio_pad|>");
    aligner_timestamp_token_id_ = aligner_tokenizer_->GetTokenId("<timestamp>");
    if (aligner_audio_start_token_id_ < 0 || aligner_audio_end_token_id_ < 0 ||
        aligner_audio_pad_token_id_ < 0 || aligner_timestamp_token_id_ < 0) {
      SHERPA_ONNX_LOGE(
          "Failed to locate <|audio_start|>/<|audio_end|>/<|audio_pad|>/"
          "<timestamp> token ids in the qwen3 forced aligner tokenizer");
      SHERPA_ONNX_EXIT(-1);
    }
  }
}

std::vector<int64_t> OfflineRecognizerQwen3ASRImpl::BuildSourceIds(
    const std::string &hotwords, const std::string &language,
    int32_t audio_token_len, int32_t *before_len,
    int32_t *fake_audio_token_len) const {
  const std::string before_utf8 = std::string(kQwen3SystemPromptPrefix) +
                                  hotwords + kQwen3SystemPromptSuffix;
  std::vector<int64_t> prompt_ids_before = tokenizer_->Encode(before_utf8);

  if (before_len) {
    *before_len = static_cast<int32_t>(prompt_ids_before.size());
  }
  if (fake_audio_token_len) {
    *fake_audio_token_len = audio_token_len;
  }

  std::vector<int64_t> prompt_ids_after_with_language;
  const std::vector<int64_t> *ids_after = &prompt_ids_after_;
  if (!language.empty()) {
    auto language_ids = tokenizer_->Encode("language " + language);
    prompt_ids_after_with_language.reserve(prompt_ids_after_.size() +
                                           language_ids.size() + 1);
    prompt_ids_after_with_language.insert(prompt_ids_after_with_language.end(),
                                          prompt_ids_after_.begin(),
                                          prompt_ids_after_.end());
    prompt_ids_after_with_language.insert(prompt_ids_after_with_language.end(),
                                          language_ids.begin(),
                                          language_ids.end());
    prompt_ids_after_with_language.push_back(asr_text_token_id_);
    ids_after = &prompt_ids_after_with_language;
  }

  std::vector<int64_t> source_ids;
  size_t estimated_size =
      prompt_ids_before.size() +
      static_cast<size_t>(audio_token_len) * audio_pad_ids_.size() +
      ids_after->size();
  source_ids.reserve(estimated_size);
  source_ids.insert(source_ids.end(), prompt_ids_before.begin(),
                    prompt_ids_before.end());

  for (int32_t i = 0; i < audio_token_len; ++i) {
    source_ids.insert(source_ids.end(), audio_pad_ids_.begin(),
                      audio_pad_ids_.end());
  }

  source_ids.insert(source_ids.end(), ids_after->begin(), ids_after->end());

  return source_ids;
}

int64_t OfflineRecognizerQwen3ASRImpl::SampleTokenFromLogitsFp16OrFp32(
    const void *logits, bool is_fp16, int32_t vocab_size) const {
  if (!logits || vocab_size <= 0) {
    return 0;
  }

  int32_t best = 0;
  float best_val = -std::numeric_limits<float>::infinity();
  bool found_valid = false;

  if (is_fp16) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = HalfBitsToFloat(p[i]);
      if (std::isfinite(v) && v > best_val) {
        best_val = v;
        best = i;
        found_valid = true;
      }
    }
  } else {
    const float *p = reinterpret_cast<const float *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = p[i];
      if (std::isfinite(v) && v > best_val) {
        best_val = v;
        best = i;
        found_valid = true;
      }
    }
  }

  return found_valid ? best : 0;
}

int64_t OfflineRecognizerQwen3ASRImpl::SampleTokenFromLogits(
    const Ort::Value &logits, int32_t time_index, float temperature,
    float top_p) const {
  auto info = logits.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() < 3 || shape[1] <= 0 || shape[2] <= 0 || time_index < 0) {
    return 0;
  }

  const int32_t time_dim = static_cast<int32_t>(shape[1]);
  if (time_index >= time_dim) {
    return 0;
  }

  const int32_t vocab_size = static_cast<int32_t>(shape[2]);
  auto elem_type =
      static_cast<ONNXTensorElementDataType>(info.GetElementType());

  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
      elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    return 0;
  }

  const bool is_fp16 = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                        elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);

  const void *base =
      is_fp16 ? static_cast<const void *>(logits.GetTensorData<uint16_t>())
              : static_cast<const void *>(logits.GetTensorData<float>());

  const size_t offset = static_cast<size_t>(time_index) * vocab_size;
  const void *row = is_fp16
                        ? static_cast<const void *>(
                              reinterpret_cast<const uint16_t *>(base) + offset)
                        : static_cast<const void *>(
                              reinterpret_cast<const float *>(base) + offset);

  return SampleTokenWithTemperatureAndTopP(row, is_fp16, vocab_size,
                                           temperature, top_p);
}

int64_t OfflineRecognizerQwen3ASRImpl::SampleTokenWithTemperatureAndTopP(
    const void *logits, bool is_fp16, int32_t vocab_size, float temperature,
    float top_p, int64_t avoid_id) const {
  if (!logits || vocab_size <= 0) {
    return 0;
  }

  if (temperature <= 1e-6f) {
    int32_t best = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    bool found_valid = false;

    if (is_fp16) {
      const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
      for (int32_t i = 0; i < vocab_size; ++i) {
        if (avoid_id >= 0 && i == avoid_id) {
          continue;
        }
        float v = HalfBitsToFloat(p[i]);
        if (std::isfinite(v) && v > best_val) {
          best_val = v;
          best = i;
          found_valid = true;
        }
      }
    } else {
      const float *p = reinterpret_cast<const float *>(logits);
      for (int32_t i = 0; i < vocab_size; ++i) {
        if (avoid_id >= 0 && i == avoid_id) {
          continue;
        }
        float v = p[i];
        if (std::isfinite(v) && v > best_val) {
          best_val = v;
          best = i;
          found_valid = true;
        }
      }
    }

    return found_valid ? best : 0;
  }

  std::vector<float> probs(vocab_size, 0.0f);
  float max_logit = -std::numeric_limits<float>::infinity();

  if (is_fp16) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      if (avoid_id >= 0 && i == avoid_id) {
        probs[i] = -std::numeric_limits<float>::infinity();
        continue;
      }

      float v = HalfBitsToFloat(p[i]);
      if (!std::isfinite(v)) {
        probs[i] = -std::numeric_limits<float>::infinity();
        continue;
      }

      probs[i] = v / temperature;
      if (probs[i] > max_logit) {
        max_logit = probs[i];
      }
    }
  } else {
    const float *p = reinterpret_cast<const float *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      if (avoid_id >= 0 && i == avoid_id) {
        probs[i] = -std::numeric_limits<float>::infinity();
        continue;
      }

      float v = p[i];
      if (!std::isfinite(v)) {
        probs[i] = -std::numeric_limits<float>::infinity();
        continue;
      }

      probs[i] = v / temperature;
      if (probs[i] > max_logit) {
        max_logit = probs[i];
      }
    }
  }

  if (!std::isfinite(max_logit)) {
    return SampleTokenFromLogitsFp16OrFp32(logits, is_fp16, vocab_size);
  }

  float sum = 0.0f;
  for (int32_t i = 0; i < vocab_size; ++i) {
    if (!std::isfinite(probs[i])) {
      probs[i] = 0.0f;
      continue;
    }

    probs[i] = std::exp(probs[i] - max_logit);
    sum += probs[i];
  }

  if (sum <= 0.0f) {
    return SampleTokenFromLogitsFp16OrFp32(logits, is_fp16, vocab_size);
  }

  if (top_p < 1.0f - 1e-6f) {
    std::vector<std::pair<int32_t, float>> prob_idx;
    prob_idx.reserve(vocab_size);

    for (int32_t i = 0; i < vocab_size; ++i) {
      if (probs[i] > 0.0f) {
        prob_idx.push_back({i, probs[i]});
      }
    }

    if (prob_idx.empty()) {
      return SampleTokenFromLogitsFp16OrFp32(logits, is_fp16, vocab_size);
    }

    std::sort(
        prob_idx.begin(), prob_idx.end(),
        [](const std::pair<int32_t, float> &a,
           const std::pair<int32_t, float> &b) { return a.second > b.second; });

    float kept_sum = 0.0f;
    int32_t cutoff = static_cast<int32_t>(prob_idx.size());
    for (int32_t i = 0; i < static_cast<int32_t>(prob_idx.size()); ++i) {
      kept_sum += prob_idx[i].second;
      if (kept_sum / sum >= top_p) {
        cutoff = i + 1;
        break;
      }
    }

    if (cutoff <= 0) {
      return prob_idx[0].first;
    }

    kept_sum = 0.0f;
    for (int32_t i = 0; i < cutoff; ++i) {
      kept_sum += prob_idx[i].second;
    }

    if (kept_sum <= 0.0f) {
      return prob_idx[0].first;
    }

    float r;
    {
      std::lock_guard<std::mutex> lock(rng_mutex_);
      r = std::uniform_real_distribution<float>(0.0f, kept_sum)(rng_);
    }
    float cumsum = 0.0f;
    for (int32_t i = 0; i < cutoff; ++i) {
      cumsum += prob_idx[i].second;
      if (r <= cumsum) {
        return prob_idx[i].first;
      }
    }

    return prob_idx[cutoff - 1].first;
  }

  float r;
  {
    std::lock_guard<std::mutex> lock(rng_mutex_);
    r = std::uniform_real_distribution<float>(0.0f, sum)(rng_);
  }
  float cumsum = 0.0f;
  for (int32_t i = 0; i < vocab_size; ++i) {
    cumsum += probs[i];
    if (r <= cumsum) {
      return i;
    }
  }

  return vocab_size - 1;
}

OfflineRecognitionResult OfflineRecognizerQwen3ASRImpl::GenerateText(
    Ort::Value audio_features, int32_t audio_token_len,
    OfflineStream *stream) const {
  OfflineRecognitionResult result;
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const auto &qwen3_config = config_.model_config.qwen3_asr;

  int32_t max_new_tokens =
      stream->GetOptionInt("max_new_tokens", qwen3_config.max_new_tokens);
  if (max_new_tokens <= 0) {
    max_new_tokens = qwen3_config.max_new_tokens;
  }

  const float temperature =
      stream->GetOptionFloat("temperature", qwen3_config.temperature);
  const float top_p = stream->GetOptionFloat("top_p", qwen3_config.top_p);

  bool all_silent = false;
  Ort::Value trimmed_audio_features = TrimAudioFeatures(
      std::move(audio_features), model_->Allocator(), &all_silent);

  auto trimmed_shape =
      trimmed_audio_features.GetTensorTypeAndShapeInfo().GetShape();
  if (trimmed_shape.size() == 3 && trimmed_shape[1] > 0) {
    audio_token_len = std::min<int32_t>(audio_token_len,
                                        static_cast<int32_t>(trimmed_shape[1]));
  }

  if (all_silent) {
    // The whole clip is silence. Force an empty result now, before any
    // hotwords/language prompt tokens are built, so they cannot bias the
    // decoder into hallucinating text for silent audio.
    audio_token_len = 0;
  }

  if (config_.model_config.debug) {
    float abs_max = TensorAbsMax(trimmed_audio_features, 1LL << 20);
    SHERPA_ONNX_LOGE(
        "qwen3-asr: audio_features shape=[%d,%d,%d] abs_max=%f "
        "audio_token_len=%d",
        static_cast<int32_t>(trimmed_shape.size() > 0 ? trimmed_shape[0] : -1),
        static_cast<int32_t>(trimmed_shape.size() > 1 ? trimmed_shape[1] : -1),
        static_cast<int32_t>(trimmed_shape.size() > 2 ? trimmed_shape[2] : -1),
        abs_max, audio_token_len);
  }

  if (audio_token_len <= 0) {
    result.text = "";
    return result;
  }

  // Optional per-stream hotwords via SetOption("hotwords", comma-separated
  // CSV).
  const std::string hotwords = Qwen3FormatHotwordsForPrompt(
      stream->HasOption("hotwords") ? stream->GetOption("hotwords")
                                    : qwen3_config.hotwords);

  std::string language;
  if (stream->HasOption("language")) {
    language = stream->GetOption("language");
  }

  int32_t before_len = 0;
  int32_t fake_audio_token_len = 0;
  std::vector<int64_t> source_ids = BuildSourceIds(
      hotwords, language, audio_token_len, &before_len, &fake_audio_token_len);

  int32_t context_len = static_cast<int32_t>(source_ids.size());
  if (context_len == 0) {
    result.text = "";
    return result;
  }

  std::vector<std::pair<Ort::Value, Ort::Value>> cache_kv =
      model_->CreateEmptyKVCache(1);
  const int32_t model_max_len = model_->GetMaxTotalLen();
  int32_t max_seq_len = model_max_len;
  const int32_t max_total_len_opt =
      stream->GetOptionInt("max_total_len", qwen3_config.max_total_len);
  if (max_total_len_opt > 0) {
    max_seq_len = std::min(model_max_len, max_total_len_opt);
  }

  if (!hotwords.empty()) {
    const std::string scaffold_no_hw =
        std::string(kQwen3SystemPromptPrefix) + kQwen3SystemPromptSuffix;
    const std::vector<int64_t> base_ids = tokenizer_->Encode(scaffold_no_hw);
    const int32_t base_before = static_cast<int32_t>(base_ids.size());
    const int32_t hotword_tokens = std::max(0, before_len - base_before);

    const int32_t tail_len = static_cast<int32_t>(prompt_ids_after_.size());
    const int32_t one_audio_len = static_cast<int32_t>(audio_pad_ids_.size());
    const int32_t room = max_seq_len - before_len - tail_len;
    const bool tight = hotword_tokens >= 48 ||
                       (one_audio_len > 0 && room < one_audio_len * 32);

    if (config_.model_config.debug || tight) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr: hotwords add %d tokenizer tokens in the prompt head "
          "(before_audio=%d, scaffold_without_hotwords=%d).",
          hotword_tokens, before_len, base_before);
    }
    if (tight) {
      Qwen3LogMaxTotalLenSuggestions(max_seq_len, model_max_len);
    }
  }

  if (context_len > max_seq_len) {
    const int32_t one_audio_len = static_cast<int32_t>(audio_pad_ids_.size());
    if (one_audio_len <= 0) {
      result.text = "";
      return result;
    }

    int32_t after_len =
        context_len - before_len - fake_audio_token_len * one_audio_len;
    if (after_len < 0) {
      after_len = 0;
    }

    int32_t keep_audio = (max_seq_len - before_len - after_len) / one_audio_len;
    if (keep_audio < 0) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr prompt scaffold exceeds max_total_len: before=%d after=%d "
          "max_total_len=%d",
          before_len, after_len, max_seq_len);
      Qwen3LogMaxTotalLenSuggestions(max_seq_len, model_max_len);
      result.text = "";
      return result;
    }

    if (keep_audio == 0) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr max_total_len=%d leaves no room for audio placeholders "
          "(before=%d after=%d)",
          max_seq_len, before_len, after_len);
      Qwen3LogMaxTotalLenSuggestions(max_seq_len, model_max_len);
      result.text = "";
      return result;
    }

    if (keep_audio < fake_audio_token_len) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr: context_len (%d) exceeds max_total_len (%d). Truncating "
          "audio placeholders: audio_token_len=%d -> keep_audio=%d (before=%d "
          "after=%d).",
          context_len, max_seq_len, fake_audio_token_len, keep_audio,
          before_len, after_len);
      Qwen3LogMaxTotalLenSuggestions(max_seq_len, model_max_len);
      std::vector<int64_t> ids_before(source_ids.begin(),
                                      source_ids.begin() + before_len);
      std::vector<int64_t> ids_after(source_ids.end() - after_len,
                                     source_ids.end());

      source_ids.clear();
      source_ids.reserve(before_len + keep_audio * one_audio_len + after_len);
      source_ids.insert(source_ids.end(), ids_before.begin(), ids_before.end());

      for (int32_t i = 0; i < keep_audio; ++i) {
        source_ids.insert(source_ids.end(), audio_pad_ids_.begin(),
                          audio_pad_ids_.end());
      }

      source_ids.insert(source_ids.end(), ids_after.begin(), ids_after.end());

      fake_audio_token_len = keep_audio;
      audio_token_len = keep_audio;
      context_len = static_cast<int32_t>(source_ids.size());

      trimmed_audio_features = TruncateAudioFeatures(
          std::move(trimmed_audio_features), keep_audio, model_->Allocator());
    }
  }

  std::vector<int64_t> input_ids = source_ids;
  std::array<int64_t, 2> ids_shape{1, context_len};
  Ort::Value input_ids_tensor =
      Ort::Value::CreateTensor(memory_info, input_ids.data(), input_ids.size(),
                               ids_shape.data(), ids_shape.size());

  std::array<int64_t, 2> attn_mask_shape{1, context_len};
  std::vector<int64_t> attn_mask_vec(context_len, 1);
  Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(
      memory_info, attn_mask_vec.data(), attn_mask_vec.size(),
      attn_mask_shape.data(), attn_mask_shape.size());

  Ort::Value cache_position =
      BuildCachePosition(model_->Allocator(), context_len);
  Ort::Value audio_features_view = View(&trimmed_audio_features);

  auto tmp = model_->ForwardLLM(
      std::move(input_ids_tensor), std::move(audio_features_view),
      std::move(attention_mask), cache_position, cache_kv);
  Ort::Value logits = std::move(tmp.first);
  auto kv_outputs = std::move(tmp.second);

  model_->ApplyKvDeltaInplace(&cache_kv, kv_outputs, cache_position);

  std::vector<int64_t> generated_ids;
  generated_ids.reserve(static_cast<size_t>(max_new_tokens));

  const int64_t eos_id = tokenizer_->GetEosTokenId();

  auto log_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
  if (log_shape.size() < 3) {
    result.text = "";
    return result;
  }

  const int32_t time_dim = static_cast<int32_t>(log_shape[1]);
  const int32_t last_idx = context_len - 1;
  if (last_idx >= time_dim) {
    if (config_.model_config.debug) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr: logits time_dim (%d) < context_len (%d); "
          "cannot sample first token",
          time_dim, context_len);
    }
    result.text = "";
    return result;
  }

  int64_t next_id = SampleTokenFromLogits(logits, last_idx, temperature, top_p);

  if (next_id == eos_id) {
    if (config_.model_config.debug) {
      float abs_max = TensorAbsMax(logits, 1LL << 20);
      SHERPA_ONNX_LOGE(
          "qwen3-asr: first token is EOS (eos_id=%d). logits_abs_max=%f "
          "context_len=%d max_total_len=%d",
          static_cast<int32_t>(eos_id), abs_max, context_len, max_seq_len);
    }

    const int32_t vocab_size = static_cast<int32_t>(log_shape[2]);
    auto elem_type = static_cast<ONNXTensorElementDataType>(
        logits.GetTensorTypeAndShapeInfo().GetElementType());
    const bool is_fp16 = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                          elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);

    const void *base =
        is_fp16 ? static_cast<const void *>(logits.GetTensorData<uint16_t>())
                : static_cast<const void *>(logits.GetTensorData<float>());

    const size_t offset = static_cast<size_t>(last_idx) * vocab_size;
    const void *row =
        is_fp16 ? static_cast<const void *>(
                      reinterpret_cast<const uint16_t *>(base) + offset)
                : static_cast<const void *>(
                      reinterpret_cast<const float *>(base) + offset);

    next_id = SampleTokenWithTemperatureAndTopP(row, is_fp16, vocab_size,
                                                temperature, top_p, eos_id);

    if (next_id == eos_id) {
      result.text = "";
      return result;
    }
  }

  generated_ids.push_back(next_id);
  int32_t cur_len = context_len;

  for (int32_t step = 1; step < max_new_tokens; ++step) {
    if (cur_len >= max_seq_len) {
      break;
    }

    if (step + 1 == max_new_tokens) {
      SHERPA_ONNX_LOGE(
          "Result is truncated. max_new_tokens %d is too small for "
          "this audio input. Please either use a shorter audio or use a "
          "larger max_new_tokens",
          max_new_tokens);
    }

    const int64_t last_token_id = next_id;
    std::vector<int64_t> one_id{last_token_id};
    std::array<int64_t, 2> one_shape{1, 1};
    Ort::Value one_tensor =
        Ort::Value::CreateTensor(memory_info, one_id.data(), one_id.size(),
                                 one_shape.data(), one_shape.size());

    std::array<int64_t, 2> mask_shape{1, 1};
    std::vector<int64_t> mask_vec(1, 1);
    Ort::Value next_attention_mask = Ort::Value::CreateTensor<int64_t>(
        memory_info, mask_vec.data(), mask_vec.size(), mask_shape.data(),
        mask_shape.size());

    std::array<int64_t, 1> cache_pos_shape{1};
    std::vector<int64_t> cache_pos_vec{static_cast<int64_t>(cur_len)};
    Ort::Value next_cache_position = Ort::Value::CreateTensor<int64_t>(
        memory_info, cache_pos_vec.data(), cache_pos_vec.size(),
        cache_pos_shape.data(), cache_pos_shape.size());

    Ort::Value audio_features_view2 = View(&trimmed_audio_features);

    auto tmp2 = model_->ForwardLLM(
        std::move(one_tensor), std::move(audio_features_view2),
        std::move(next_attention_mask), next_cache_position, cache_kv);
    logits = std::move(tmp2.first);
    auto kv_outputs2 = std::move(tmp2.second);

    model_->ApplyKvDeltaInplace(&cache_kv, kv_outputs2, next_cache_position);

    auto log_shape2 = logits.GetTensorTypeAndShapeInfo().GetShape();
    if (log_shape2.size() < 3) {
      break;
    }

    const int32_t time_dim2 = static_cast<int32_t>(log_shape2[1]);
    if (time_dim2 < 1) {
      break;
    }

    next_id = SampleTokenFromLogits(logits, time_dim2 - 1, temperature, top_p);

    if (next_id == eos_id) {
      break;
    }

    generated_ids.push_back(next_id);
    ++cur_len;

    if (IsDegenerateRepetition(generated_ids)) {
      // The decoder has collapsed into cycling over a couple of token ids;
      // it will not recover under greedy decoding and would only repeat
      // them until max_new_tokens. Stop, and drop the repetition so the
      // caller sees the text decoded before the collapse instead of a
      // window full of one syllable. Seen with the Qwen3-ASR 1.7B model on
      // some sung inputs; see k2-fsa/sherpa-onnx#3535.
      SHERPA_ONNX_LOGE(
          "qwen3-asr: decode collapsed into a repetition loop after %d "
          "tokens; truncating the repetition",
          static_cast<int32_t>(generated_ids.size()));
      TrimDegenerateTail(&generated_ids);
      break;
    }
  }

  std::vector<int64_t> cleaned_ids = generated_ids;
  if (!generated_ids.empty()) {
    const size_t prefix_window = std::min<size_t>(16, generated_ids.size());
    auto asr_text_it =
        std::find(generated_ids.begin(), generated_ids.begin() + prefix_window,
                  asr_text_token_id_);

    // Only strip a leading scaffold prefix recognized by token ID.
    if (asr_text_it != generated_ids.begin() + prefix_window &&
        asr_text_it != generated_ids.begin()) {
      std::vector<int64_t> prefix_ids(generated_ids.begin(),
                                      std::next(asr_text_it));
      std::string prefix_text = tokenizer_->Decode(prefix_ids);
      if (prefix_text.rfind("language ", 0) == 0 && prefix_text.size() >= 10 &&
          prefix_text.compare(prefix_text.size() - 10, 10, "<asr_text>") == 0) {
        // "language Chinese<asr_text>" -> "Chinese"
        result.lang = prefix_text.substr(9, prefix_text.size() - 9 - 10);
        cleaned_ids.assign(std::next(asr_text_it), generated_ids.end());
      }
    }
  }

  if (result.lang.empty() && !language.empty()) {
    result.lang = language;
  }

  result.text = tokenizer_->Decode(cleaned_ids);
  RemoveUtf8ReplacementChars(&result.text);

  if (!cleaned_ids.empty()) {
    std::vector<std::string> all_tokens;
    all_tokens.reserve(cleaned_ids.size());
    std::string pending_bytes;

    for (int64_t token_id : cleaned_ids) {
      std::string s =
          tokenizer_->GetTokenStringStreaming(token_id, &pending_bytes);
      all_tokens.push_back(std::move(s));
    }

    if (!pending_bytes.empty() && !all_tokens.empty()) {
      all_tokens.back().append("\xEF\xBF\xBD");
    }

    result.tokens = std::move(all_tokens);
  }

  return result;
}

bool OfflineRecognizerQwen3ASRImpl::RunForcedAlignment(
    const std::vector<float> &mel_features, int32_t feat_frames,
    OfflineRecognitionResult *r) const {
  if (!aligner_model_ || r->text.empty()) {
    return false;
  }

  std::vector<std::string> split_words = SplitQwen3AlignerWords(r->text);
  if (split_words.empty()) {
    return false;
  }

  // Pre-encode so words that produce no tokens don't create orphaned
  // timestamp slots.
  std::vector<std::string> words;
  std::vector<std::vector<int64_t>> word_ids;
  for (auto &word : split_words) {
    std::vector<int64_t> wids = aligner_tokenizer_->Encode(word);
    if (!wids.empty()) {
      words.push_back(std::move(word));
      word_ids.push_back(std::move(wids));
    }
  }
  if (words.empty()) {
    return false;
  }

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::array<int64_t, 3> conv_input_shape{1, static_cast<int64_t>(feat_frames),
                                          static_cast<int64_t>(kQwen3MelDim)};
  Ort::Value conv_input = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(mel_features.data()),
      static_cast<size_t>(feat_frames) * kQwen3MelDim, conv_input_shape.data(),
      conv_input_shape.size());

  Ort::Value conv_output =
      aligner_model_->ForwardConvFrontend(std::move(conv_input));

  auto conv_shape = conv_output.GetTensorTypeAndShapeInfo().GetShape();
  if (conv_shape.size() < 3 || conv_shape[1] <= 0) {
    return false;
  }

  const int32_t conv_num_frames = static_cast<int32_t>(conv_shape[1]);
  const int32_t expected_audio_token_len =
      FeatToAudioTokensLen(feat_frames, kQwen3ChunkSize);
  const int32_t valid_frames =
      std::min(expected_audio_token_len, conv_num_frames);
  if (valid_frames <= 0) {
    return false;
  }

  // The aligner classifies each timestamp slot into 5000 buckets of 80 ms,
  // i.e. it cannot represent audio beyond ~400 s in a single pass.
  if (valid_frames > 5000) {
    SHERPA_ONNX_LOGE(
        "qwen3-forced-aligner: audio is too long for a single alignment pass "
        "(%d audio tokens > 5000); skipping timestamps",
        valid_frames);
    return false;
  }

  auto mask_buf =
      std::make_unique<bool[]>(static_cast<size_t>(conv_num_frames));
  std::fill_n(mask_buf.get(), static_cast<size_t>(valid_frames), true);

  std::array<int64_t, 2> tok_mask_shape{1, conv_num_frames};
  Ort::Value feature_attention_mask = Ort::Value::CreateTensor<bool>(
      memory_info, mask_buf.get(), static_cast<size_t>(conv_num_frames),
      tok_mask_shape.data(), tok_mask_shape.size());

  Ort::Value audio_features = aligner_model_->ForwardEncoder(
      std::move(conv_output), std::move(feature_attention_mask));

  audio_features = TruncateAudioFeatures(
      std::move(audio_features), valid_frames, aligner_model_->Allocator());

  // Sequence layout (mirrors encode_timestamp() in qwen3_forced_aligner.py):
  //   <|audio_start|> <|audio_pad|>*A <|audio_end|>
  //   word_0 <timestamp> <timestamp> word_1 <timestamp> <timestamp> ...
  // where word_i's start/end times are predicted by the pair of <timestamp>
  // slots following its tokens.
  std::vector<int64_t> ids;
  ids.reserve(static_cast<size_t>(valid_frames) + 2 * words.size() + 16);
  ids.push_back(aligner_audio_start_token_id_);
  ids.insert(ids.end(), static_cast<size_t>(valid_frames),
             aligner_audio_pad_token_id_);
  ids.push_back(aligner_audio_end_token_id_);

  for (const auto &wids : word_ids) {
    ids.insert(ids.end(), wids.begin(), wids.end());
    ids.push_back(aligner_timestamp_token_id_);
    ids.push_back(aligner_timestamp_token_id_);
  }

  std::array<int64_t, 2> ids_shape{1, static_cast<int64_t>(ids.size())};
  Ort::Value input_ids = Ort::Value::CreateTensor<int64_t>(
      memory_info, ids.data(), ids.size(), ids_shape.data(), ids_shape.size());

  std::vector<int64_t> mask_vec(ids.size(), 1);
  Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(
      memory_info, mask_vec.data(), mask_vec.size(), ids_shape.data(),
      ids_shape.size());

  Ort::Value logits = aligner_model_->ForwardDecoder(std::move(input_ids),
                                                     std::move(audio_features),
                                                     std::move(attention_mask));

  auto logits_info = logits.GetTensorTypeAndShapeInfo();
  auto logits_shape = logits_info.GetShape();
  if (logits_shape.size() != 3 || logits_shape[2] <= 0) {
    SHERPA_ONNX_LOGE("qwen3-forced-aligner: unexpected logits rank %d",
                     static_cast<int32_t>(logits_shape.size()));
    return false;
  }

  auto logits_elem_type =
      static_cast<ONNXTensorElementDataType>(logits_info.GetElementType());
  if (!IsFloatOrHalfBitsTensorType(logits_elem_type)) {
    SHERPA_ONNX_LOGE("qwen3-forced-aligner: unsupported logits element type %d",
                     static_cast<int32_t>(logits_elem_type));
    return false;
  }

  const int32_t seq_len = static_cast<int32_t>(logits_shape[1]);
  const int32_t num_classes = static_cast<int32_t>(logits_shape[2]);
  const float *logits_f32 = nullptr;
  const uint16_t *logits_f16_bits = nullptr;
  if (logits_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    logits_f32 = logits.GetTensorData<float>();
  } else {
    logits_f16_bits = logits.GetTensorData<uint16_t>();
  }

  std::vector<int64_t> raw_indices;
  std::vector<int32_t> ts_positions;
  for (int32_t i = 0; i < seq_len && i < static_cast<int32_t>(ids.size());
       ++i) {
    if (ids[i] != aligner_timestamp_token_id_) {
      continue;
    }
    const int64_t row_offset = static_cast<int64_t>(i) * num_classes;
    int32_t argmax = 0;
    float best = ReadFloatOrHalfBitsValue(logits_f32, logits_f16_bits,
                                          logits_elem_type, row_offset);
    for (int32_t c = 1; c < num_classes; ++c) {
      float v = ReadFloatOrHalfBitsValue(logits_f32, logits_f16_bits,
                                         logits_elem_type, row_offset + c);
      if (v > best) {
        best = v;
        argmax = c;
      }
    }
    ts_positions.push_back(argmax);
  }

  if (ts_positions.size() != words.size() * 2) {
    SHERPA_ONNX_LOGE(
        "qwen3-forced-aligner: expected %d timestamp slots, got %d; "
        "skipping timestamps",
        static_cast<int32_t>(words.size() * 2),
        static_cast<int32_t>(ts_positions.size()));
    return false;
  }

  raw_indices.assign(ts_positions.begin(), ts_positions.end());
  FixQwen3AlignerTimestamps(&raw_indices);

  // Each index unit is 80 ms (timestamp_segment_time in the model config).
  constexpr float kSecondsPerIndex = 0.08f;

  std::vector<std::string> out_tokens;
  std::vector<float> out_timestamps;
  std::vector<float> out_durations;
  out_tokens.reserve(words.size());
  out_timestamps.reserve(words.size());
  out_durations.reserve(words.size());

  for (size_t i = 0; i < words.size(); ++i) {
    const float start = raw_indices[2 * i] * kSecondsPerIndex;
    const float end = raw_indices[2 * i + 1] * kSecondsPerIndex;
    out_tokens.push_back(words[i]);
    out_timestamps.push_back(start);
    out_durations.push_back(std::max(0.0f, end - start));
  }

  r->tokens = std::move(out_tokens);
  r->timestamps = std::move(out_timestamps);
  r->durations = std::move(out_durations);

  return true;
}

void OfflineRecognizerQwen3ASRImpl::DecodeStreams(OfflineStream **ss,
                                                  int32_t n) const {
  for (int32_t i = 0; i != n; ++i) {
    Decode(ss[i]);
  }
}

void OfflineRecognizerQwen3ASRImpl::Decode(OfflineStream *stream) const {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::vector<float> f = stream->GetFrames();
  if (f.empty()) {
    OfflineRecognitionResult r;
    r.text = "";
    stream->SetResult(r);
    return;
  }

  int32_t num_frames =
      static_cast<int32_t>(f.size() / static_cast<size_t>(kQwen3MelDim));
  if (static_cast<size_t>(num_frames) * static_cast<size_t>(kQwen3MelDim) !=
      f.size()) {
    OfflineRecognitionResult r;
    r.text = "";
    stream->SetResult(r);
    return;
  }
  if (num_frames < 2) {
    OfflineRecognitionResult r;
    r.text = "";
    stream->SetResult(r);
    return;
  }

  NormalizeWhisperFeatures(f.data(), num_frames, kQwen3MelDim);

  int32_t F = kQwen3MelDim;
  int32_t feat_frames = num_frames;

  std::array<int64_t, 3> conv_input_shape{1, static_cast<int64_t>(feat_frames),
                                          static_cast<int64_t>(F)};

  Ort::Value conv_input = Ort::Value::CreateTensor<float>(
      memory_info, f.data(), static_cast<size_t>(feat_frames) * F,
      conv_input_shape.data(), conv_input_shape.size());

  Ort::Value conv_output = model_->ForwardConvFrontend(std::move(conv_input));

  auto conv_shape = conv_output.GetTensorTypeAndShapeInfo().GetShape();
  if (conv_shape.size() < 3 || conv_shape[1] <= 0) {
    OfflineRecognitionResult r;
    r.text = "";
    stream->SetResult(r);
    return;
  }

  int32_t conv_num_frames = static_cast<int32_t>(conv_shape[1]);
  int32_t expected_audio_token_len =
      FeatToAudioTokensLen(feat_frames, kQwen3ChunkSize);

  int32_t valid_frames = std::min(expected_audio_token_len, conv_num_frames);
  auto mask_buf =
      std::make_unique<bool[]>(static_cast<size_t>(conv_num_frames));
  std::fill_n(mask_buf.get(), static_cast<size_t>(valid_frames), true);

  std::array<int64_t, 2> tok_mask_shape{1, conv_num_frames};
  Ort::Value feature_attention_mask = Ort::Value::CreateTensor<bool>(
      memory_info, mask_buf.get(), static_cast<size_t>(conv_num_frames),
      tok_mask_shape.data(), tok_mask_shape.size());

  Ort::Value audio_features = model_->ForwardEncoder(
      std::move(conv_output), std::move(feature_attention_mask));

  if (config_.model_config.debug) {
    SHERPA_ONNX_LOGE(
        "qwen3-asr: feat_frames=%d conv_frames=%d expected_audio_tokens=%d "
        "valid_audio_tokens=%d",
        feat_frames, conv_num_frames, expected_audio_token_len, valid_frames);
  }

  OfflineRecognitionResult r =
      GenerateText(std::move(audio_features), valid_frames, stream);

  r.text = ApplyHomophoneReplacer(std::move(r.text));

  if (aligner_model_ && !r.text.empty()) {
    RunForcedAlignment(f, feat_frames, &r);
  }

  stream->SetResult(r);
}

#if __ANDROID_API__ >= 9
template OfflineRecognizerQwen3ASRImpl::OfflineRecognizerQwen3ASRImpl(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizerQwen3ASRImpl::OfflineRecognizerQwen3ASRImpl(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx
