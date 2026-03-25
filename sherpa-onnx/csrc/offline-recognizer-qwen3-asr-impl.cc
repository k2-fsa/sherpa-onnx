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

Ort::Value TrimAudioFeatures(Ort::Value audio_features,
                             OrtAllocator *allocator) {
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
  InitPromptTemplateIds();
}

std::unique_ptr<OfflineStream> OfflineRecognizerQwen3ASRImpl::CreateStream()
    const {
  return std::make_unique<OfflineStream>(WhisperTag{kQwen3MelDim});
}

void OfflineRecognizerQwen3ASRImpl::InitPromptTemplateIds() {
  const std::string system_text = "<|im_start|>system\n<|im_end|>\n";
  const std::string user_prefix = "<|im_start|>user\n<|audio_start|>";
  const std::string audio_pad = "<|audio_pad|>";
  const std::string user_suffix = "<|audio_end|><|im_end|>\n";
  const std::string assistant_text = "<|im_start|>assistant\n";

  prompt_ids_before_ = tokenizer_->Encode(system_text + user_prefix);
  audio_pad_ids_ = tokenizer_->Encode(audio_pad);
  prompt_ids_after_ = tokenizer_->Encode(user_suffix + assistant_text);

  if (audio_pad_ids_.empty()) {
    SHERPA_ONNX_LOGE("Failed to tokenize <|audio_pad|> for qwen3-asr prompt");
    SHERPA_ONNX_EXIT(-1);
  }
}

std::vector<int64_t> OfflineRecognizerQwen3ASRImpl::BuildSourceIds(
    int32_t audio_token_len, int32_t *before_len,
    int32_t *fake_audio_token_len) const {
  if (before_len) {
    *before_len = static_cast<int32_t>(prompt_ids_before_.size());
  }
  if (fake_audio_token_len) {
    *fake_audio_token_len = audio_token_len;
  }

  std::vector<int64_t> source_ids;
  size_t estimated_size =
      prompt_ids_before_.size() +
      static_cast<size_t>(audio_token_len) * audio_pad_ids_.size() +
      prompt_ids_after_.size();
  source_ids.reserve(estimated_size);
  source_ids.insert(source_ids.end(), prompt_ids_before_.begin(),
                    prompt_ids_before_.end());

  for (int32_t i = 0; i < audio_token_len; ++i) {
    source_ids.insert(source_ids.end(), audio_pad_ids_.begin(),
                      audio_pad_ids_.end());
  }

  source_ids.insert(source_ids.end(), prompt_ids_after_.begin(),
                    prompt_ids_after_.end());

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

    float r = std::uniform_real_distribution<float>(0.0f, kept_sum)(rng_);
    float cumsum = 0.0f;
    for (int32_t i = 0; i < cutoff; ++i) {
      cumsum += prob_idx[i].second;
      if (r <= cumsum) {
        return prob_idx[i].first;
      }
    }

    return prob_idx[cutoff - 1].first;
  }

  float r = std::uniform_real_distribution<float>(0.0f, sum)(rng_);
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

  Ort::Value trimmed_audio_features =
      TrimAudioFeatures(std::move(audio_features), model_->Allocator());

  auto trimmed_shape =
      trimmed_audio_features.GetTensorTypeAndShapeInfo().GetShape();
  if (trimmed_shape.size() == 3 && trimmed_shape[1] > 0) {
    audio_token_len = std::min<int32_t>(audio_token_len,
                                        static_cast<int32_t>(trimmed_shape[1]));
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

  int32_t before_len = 0;
  int32_t fake_audio_token_len = 0;
  std::vector<int64_t> source_ids =
      BuildSourceIds(audio_token_len, &before_len, &fake_audio_token_len);

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
      result.text = "";
      return result;
    }

    if (keep_audio == 0) {
      SHERPA_ONNX_LOGE(
          "qwen3-asr max_total_len=%d leaves no room for audio placeholders "
          "(before=%d after=%d)",
          max_seq_len, before_len, after_len);
      result.text = "";
      return result;
    }

    if (keep_audio < fake_audio_token_len) {
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
  }

  // drop the first 3 tokens which contain things like:
  // language None<asr_text>
  if (generated_ids.size() >= 3) {
    generated_ids.erase(generated_ids.begin(), generated_ids.begin() + 3);
  }

  result.text = tokenizer_->Decode(generated_ids);
  RemoveUtf8ReplacementChars(&result.text);

  if (!generated_ids.empty()) {
    result.tokens.reserve(generated_ids.size());
    std::string pending_bytes;

    for (int64_t token_id : generated_ids) {
      std::string s =
          tokenizer_->GetTokenStringStreaming(token_id, &pending_bytes);
      result.tokens.push_back(std::move(s));
    }

    if (!pending_bytes.empty() && !result.tokens.empty()) {
      result.tokens.back().append("\xEF\xBF\xBD");
    }
  }

  return result;
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
