// sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.cc

#include "sherpa-onnx/csrc/offline-recognizer-funasr-nano-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// Convert IEEE 754 half-precision (16-bit) float to single-precision (32-bit)
// float. Handles special cases: zero, subnormal, normal, infinity, and NaN.
static inline float Fp16ToFp32(uint16_t h) {
  const uint32_t s = (h >> 15) & 0x1;
  const uint32_t e = (h >> 10) & 0x1F;
  const uint32_t f = h & 0x3FF;
  if (e == 0) {
    if (f == 0) {
      uint32_t bits = s << 31;
      float out;
      std::memcpy(&out, &bits, sizeof(out));
      return out;
    }
    float mant = static_cast<float>(f) / 1024.0f;
    float val = std::ldexp(mant, -14);
    return s ? -val : val;
  }
  if (e == 31) {
    uint32_t bits = (s << 31) | 0x7F800000u | (f << 13);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
  }
  uint32_t bits = (s << 31) | ((e + (127 - 15)) << 23) | (f << 13);
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// Convert IEEE 754 single-precision (32-bit) float to half-precision (16-bit)
// float. Handles overflow (clamped to infinity), underflow (clamped to zero),
// and normal values with proper rounding.
static inline uint16_t Fp32ToFp16(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  uint32_t sign = (bits >> 31) & 1;
  int32_t exp = ((bits >> 23) & 0xFF) - 127;
  uint32_t mant = bits & 0x7FFFFF;
  if (exp > 15) {
    return static_cast<uint16_t>((sign << 15) | (0x1F << 10));
  }
  if (exp < -14) {
    if (exp < -24) {
      return static_cast<uint16_t>(sign << 15);
    }
    mant |= 0x800000;
    int shift = (-14 - exp);
    uint16_t sub = static_cast<uint16_t>(mant >> (shift + 13));
    return static_cast<uint16_t>((sign << 15) | sub);
  }
  uint16_t he = static_cast<uint16_t>(exp + 15);
  uint16_t hm = static_cast<uint16_t>(mant >> 13);
  return static_cast<uint16_t>((sign << 15) | (he << 10) | hm);
}

}  // namespace

OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(config),
      config_(config),
      model_(std::make_unique<OfflineFunASRNanoModel>(config.model_config)),
      tokenizer_(std::make_unique<FunASRNanoTokenizer>(
          config.model_config.funasr_nano.tokenizer)),
      rng_(config.model_config.funasr_nano.seed) {
  InitFeatConfig();
}

template <typename Manager>
OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    Manager *mgr, const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(mgr, config),
      config_(config),
      model_(std::make_unique<OfflineFunASRNanoModel>(mgr, config.model_config)),
      tokenizer_(std::make_unique<FunASRNanoTokenizer>(
          mgr, config.model_config.funasr_nano.tokenizer)),
      rng_(config.model_config.funasr_nano.seed) {
  InitFeatConfig();
}

std::unique_ptr<OfflineStream>
OfflineRecognizerFunASRNanoImpl::CreateStream() const {
  return std::make_unique<OfflineStream>(config_.feat_config);
}

// Initialize feature extraction configuration for FunASR-nano.
// Sets normalization, window type, and disables edge snipping and dithering
// to match the model's expected input format.
void OfflineRecognizerFunASRNanoImpl::InitFeatConfig() {
  config_.feat_config.normalize_samples = true;
  config_.feat_config.window_type = "hamming";
  config_.feat_config.snip_edges = false;
  config_.feat_config.dither = 0.0f;
}

// Apply Low Frame Rate (LFR) processing to reduce temporal resolution.
// Concatenates multiple consecutive frames into a single frame.
std::vector<float> OfflineRecognizerFunASRNanoImpl::ApplyLFR(
    const std::vector<float> &in) const {
  int32_t lfr_window_size = model_->LfrWindowSize();
  int32_t lfr_window_shift = model_->LfrWindowShift();
  int32_t in_feat_dim = config_.feat_config.feature_dim;
  int32_t in_num_frames = static_cast<int32_t>(in.size() / in_feat_dim);
  int32_t out_num_frames =
      (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
  if (out_num_frames <= 0) return {};
  int32_t out_feat_dim = in_feat_dim * lfr_window_size;
  std::vector<float> out(out_num_frames * out_feat_dim);
  const float *p_in = in.data();
  float *p_out = out.data();
  for (int32_t i = 0; i != out_num_frames; ++i) {
    std::copy(p_in, p_in + out_feat_dim, p_out);
    p_out += out_feat_dim;
    p_in += lfr_window_shift * in_feat_dim;
  }
  return out;
}

// Build source token IDs with chat template format:
// [system_prompt] [user_prompt] [audio_tokens] [assistant_prompt]
// Returns the token sequence and sets fbank_beg_idx to the start position
// of audio tokens in the sequence.
std::vector<int64_t> OfflineRecognizerFunASRNanoImpl::BuildSourceIds(
    const std::string &system_prompt, const std::string &user_prompt,
    int32_t audio_token_len, int32_t &fbank_beg_idx,
    int32_t &fake_token_len) const {
  const std::string system_text =
      "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
  const std::string user_text = "<|im_start|>user\n" + user_prompt;
  const std::string after_text = "<|im_end|>\n<|im_start|>assistant\n";
  std::vector<int64_t> ids_before =
      tokenizer_->Encode(system_text + user_text);
  std::vector<int64_t> ids_after = tokenizer_->Encode(after_text);
  fbank_beg_idx = static_cast<int32_t>(ids_before.size());
  fake_token_len = audio_token_len;
  int64_t pad_id = tokenizer_->GetPadTokenId();
  if (pad_id < 0) pad_id = tokenizer_->GetEosTokenId();
  std::vector<int64_t> source_ids;
  source_ids.reserve(ids_before.size() + audio_token_len + ids_after.size());
  source_ids.insert(source_ids.end(), ids_before.begin(), ids_before.end());
  // Use pad tokens as placeholders for audio embeddings
  source_ids.insert(source_ids.end(), audio_token_len, pad_id);
  source_ids.insert(source_ids.end(), ids_after.begin(), ids_after.end());
  return source_ids;
}

// Sample token from logits using greedy decoding (argmax).
// Handles both FP16 and FP32 logits, skipping NaN/Inf values.
// Returns token ID 0 as fallback if all logits are invalid.
int64_t OfflineRecognizerFunASRNanoImpl::SampleTokenFromLogitsFp16OrFp32(
    const void *logits, bool is_fp16, int32_t vocab_size) const {
  int32_t best = 0;
  float best_val = -1e30f;
  bool found_valid = false;
  if (is_fp16) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      float v = Fp16ToFp32(p[i]);
      if (std::isfinite(v) && v > best_val) {
        best_val = v;
        best = i;
        found_valid = true;
      }
    }
  } else {
    const float *p = reinterpret_cast<const float *>(logits);
    for (int32_t i = 0; i < vocab_size; ++i) {
      if (std::isfinite(p[i]) && p[i] > best_val) {
        best_val = p[i];
        best = i;
        found_valid = true;
      }
    }
  }
  if (!found_valid) {
    return 0;
  }
  return static_cast<int64_t>(best);
}

// Generate text from encoder output using autoregressive decoding with KV cache.
// Combines text embeddings (from prompts) and audio embeddings (from encoder)
// to form the input sequence, then generates tokens autoregressively.
OfflineRecognitionResult OfflineRecognizerFunASRNanoImpl::GenerateText(
    Ort::Value encoder_out, const std::string &system_prompt,
    const std::string &user_prompt) const {
  OfflineRecognitionResult result;
  if (!model_->HasEmbeddingModel()) {
    SHERPA_ONNX_LOGE("Embedding model is required but not provided.");
    result.text = "";
    return result;
  }
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const auto &funasr_config = config_.model_config.funasr_nano;
  auto enc_shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t audio_token_len = static_cast<int32_t>(enc_shape[1]);
  int32_t hidden_size = static_cast<int32_t>(enc_shape[2]);
  int32_t fbank_beg_idx = 0;
  int32_t fake_token_len = 0;
  std::vector<int64_t> source_ids = BuildSourceIds(
      system_prompt, user_prompt, audio_token_len, fbank_beg_idx,
      fake_token_len);
  int32_t context_len = static_cast<int32_t>(source_ids.size());
  const int32_t max_seq_len = 2048;
  if (context_len > max_seq_len) {
    source_ids.resize(max_seq_len);
    context_len = max_seq_len;
  }
  // Get text embeddings for the prompt tokens
  std::vector<int64_t> input_ids = source_ids;
  std::array<int64_t, 2> ids_shape{1, context_len};
  Ort::Value input_ids_tensor = Ort::Value::CreateTensor(
      memory_info, input_ids.data(), input_ids.size(), ids_shape.data(),
      ids_shape.size());
  Ort::Value text_embeds =
      model_->ForwardEmbedding(std::move(input_ids_tensor));
  auto te_info = text_embeds.GetTensorTypeAndShapeInfo();
  auto te_shape = te_info.GetShape();
  if (static_cast<int32_t>(te_shape[2]) != hidden_size) {
    SHERPA_ONNX_LOGE("Embedding hidden mismatch: %d vs %d",
                     static_cast<int32_t>(te_shape[2]), hidden_size);
    result.text = "";
    return result;
  }
  const auto te_type = te_info.GetElementType();
  const bool te_fp16 = (te_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  // Pre-allocate full inputs_embeds buffer to max_seq_len
  std::vector<uint16_t> inputs_embeds_fp16(
      static_cast<size_t>(max_seq_len) * hidden_size, Fp32ToFp16(0.0f));
  std::vector<int64_t> attention_mask(static_cast<size_t>(max_seq_len), 0);
  // Copy text embeddings into inputs_embeds buffer
  if (te_fp16) {
    const uint16_t *p = text_embeds.GetTensorData<uint16_t>();
    std::copy(p, p + static_cast<size_t>(context_len) * hidden_size,
              inputs_embeds_fp16.data());
  } else {
    const float *p = text_embeds.GetTensorData<float>();
    for (int64_t i = 0; i < static_cast<int64_t>(context_len) * hidden_size;
         ++i) {
      inputs_embeds_fp16[static_cast<size_t>(i)] = Fp32ToFp16(p[i]);
    }
  }
  // Inject audio embeddings into inputs_embeds at the position of audio tokens
  auto enc_info2 = encoder_out.GetTensorTypeAndShapeInfo();
  auto enc_et =
      static_cast<ONNXTensorElementDataType>(enc_info2.GetElementType());
  int32_t copy_len = std::min(fake_token_len, audio_token_len);
  if (enc_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const uint16_t *enc = encoder_out.GetTensorData<uint16_t>();
    for (int32_t t = 0; t < copy_len; ++t) {
      const uint16_t *src = enc + static_cast<int64_t>(t) * hidden_size;
      uint16_t *dst = inputs_embeds_fp16.data() +
                      static_cast<int64_t>(fbank_beg_idx + t) * hidden_size;
      std::copy(src, src + hidden_size, dst);
    }
  } else if (enc_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float *enc = encoder_out.GetTensorData<float>();
    for (int32_t t = 0; t < copy_len; ++t) {
      const float *src = enc + static_cast<int64_t>(t) * hidden_size;
      uint16_t *dst = inputs_embeds_fp16.data() +
                      static_cast<int64_t>(fbank_beg_idx + t) * hidden_size;
      for (int32_t d = 0; d < hidden_size; ++d) {
        dst[d] = Fp32ToFp16(src[d]);
      }
    }
  } else {
    SHERPA_ONNX_LOGE("encoder_out elem_type=%d not supported", (int)enc_et);
    result.text = "";
    return result;
  }
  // Set attention mask for context tokens
  for (int32_t i = 0; i < context_len; ++i) attention_mask[i] = 1;
  int32_t valid_len = context_len;
  std::vector<int64_t> generated_ids;
  
  generated_ids.reserve(funasr_config.max_new_tokens);
  const int64_t eos_id = tokenizer_->GetEosTokenId();
  const int64_t im_end_id = tokenizer_->GetImEndTokenId();
  const int32_t max_new_tokens =
      funasr_config.max_new_tokens > 0 ? funasr_config.max_new_tokens : 256;
  std::vector<std::pair<Ort::Value, Ort::Value>> past_key_values;
  bool is_first_step = true;
  // Autoregressive generation loop
  for (int32_t step = 0; step < max_new_tokens; ++step) {
    if (valid_len >= max_seq_len) break;
    Ort::Value logits(nullptr);
    
    if (is_first_step) {
      // First step: use prefill model with full context
      std::array<int64_t, 3> embeds_shape{1, context_len, hidden_size};
      Ort::Value inputs_embeds_tensor = Ort::Value::CreateTensor(
          memory_info, inputs_embeds_fp16.data(),
          static_cast<size_t>(context_len) * hidden_size * sizeof(uint16_t),
          embeds_shape.data(), embeds_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

      std::array<int64_t, 2> mask_shape{1, context_len};
      Ort::Value attention_mask_tensor =
          Ort::Value::CreateTensor<int64_t>(
              memory_info, attention_mask.data(),
              static_cast<size_t>(context_len), mask_shape.data(),
              mask_shape.size());
      auto tmp = model_->ForwardLLMPrefill(std::move(inputs_embeds_tensor),
                                           std::move(attention_mask_tensor));
      logits = std::move(tmp.first);
      past_key_values = std::move(tmp.second);
    } else {
      // Subsequent steps: use decode model with KV cache
      int64_t last_token_id = generated_ids.back();
      std::vector<int64_t> one_id{last_token_id};
      std::array<int64_t, 2> one_shape{1, 1};
      Ort::Value one_tensor = Ort::Value::CreateTensor(
          memory_info, one_id.data(), one_id.size(), one_shape.data(),
          one_shape.size());
      Ort::Value next_embed =
          model_->ForwardEmbedding(std::move(one_tensor));
      auto ne_info = next_embed.GetTensorTypeAndShapeInfo();
      bool ne_fp16 = (ne_info.GetElementType() ==
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
      std::vector<uint16_t> next_embed_fp16(hidden_size);
      if (ne_fp16) {
        const uint16_t *src = next_embed.GetTensorData<uint16_t>();
        std::copy(src, src + hidden_size, next_embed_fp16.data());
      } else {
        const float *src = next_embed.GetTensorData<float>();
        for (int32_t d = 0; d < hidden_size; ++d) {
          next_embed_fp16[d] = Fp32ToFp16(src[d]);
        }
      }
      std::array<int64_t, 3> embeds_shape{1, 1, hidden_size};
      Ort::Value inputs_embeds_tensor = Ort::Value::CreateTensor(
          memory_info, next_embed_fp16.data(),
          static_cast<size_t>(hidden_size) * sizeof(uint16_t),
          embeds_shape.data(), embeds_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
      std::vector<int64_t> decode_mask(valid_len, 1);
      std::array<int64_t, 2> mask_shape{1, valid_len};
      Ort::Value attention_mask_tensor =
          Ort::Value::CreateTensor<int64_t>(
              memory_info, decode_mask.data(), decode_mask.size(),
              mask_shape.data(), mask_shape.size());
      auto tmp = model_->ForwardLLMDecode(std::move(inputs_embeds_tensor),
                                          std::move(attention_mask_tensor),
                                          past_key_values);
      logits = std::move(tmp.first);
      past_key_values = std::move(tmp.second);
    }
    // Extract logits for the last position
    auto log_info = logits.GetTensorTypeAndShapeInfo();
    auto log_shape = log_info.GetShape();
    int32_t vocab_size = static_cast<int32_t>(log_shape[2]);
    const bool log_fp16 =
        (log_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    // In KV cache mode: prefill uses last position, decode uses position 0
    int32_t last_idx = is_first_step ? (context_len - 1) : 0;
    const void *base = nullptr;
    if (log_fp16) {
      base = logits.GetTensorData<uint16_t>();
    } else {
      base = logits.GetTensorData<float>();
    }
    const size_t offset = static_cast<size_t>(last_idx) * vocab_size;
    const void *last_logits =
        log_fp16
            ? static_cast<const void *>(
                  reinterpret_cast<const uint16_t *>(base) + offset)
            : static_cast<const void *>(
                  reinterpret_cast<const float *>(base) + offset);
    // Sample next token using greedy decoding
    int64_t next_id =
        SampleTokenFromLogitsFp16OrFp32(last_logits, log_fp16, vocab_size);
    if (next_id == eos_id || next_id == im_end_id) {
      break;
    }
    generated_ids.push_back(next_id);

    // After sampling the first token from prefill, switch to decode mode
    if (is_first_step) {
      is_first_step = false;
    }
    valid_len += 1;
  }
  // Decode generated token IDs to text
  result.text = tokenizer_->Decode(generated_ids);
  result.text = ApplyInverseTextNormalization(std::move(result.text));
  result.text = ApplyHomophoneReplacer(std::move(result.text));
  if (!generated_ids.empty()) {
    // Fill tokens: decode each token individually
    result.tokens.reserve(generated_ids.size());
    for (int64_t token_id : generated_ids) {
      std::vector<int64_t> single_token{token_id};
      std::string token_str = tokenizer_->Decode(single_token);
      result.tokens.push_back(token_str);
    }
    // Fill timestamps: estimate based on audio duration and token count
    auto enc_shape2 = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
    int32_t audio_token_len2 = static_cast<int32_t>(enc_shape2[1]);
    int32_t lfr_window_size = model_->LfrWindowSize();
    int32_t lfr_window_shift = model_->LfrWindowShift();
    int32_t sampling_rate = config_.feat_config.sampling_rate;
    // Reverse LFR to get original feature frames
    int32_t original_feature_frames =
        (audio_token_len2 > 0)
            ? ((audio_token_len2 - 1) * lfr_window_shift + lfr_window_size)
            : 0;
    float frame_shift_ms = config_.feat_config.frame_shift_ms;
    float audio_duration =
        (original_feature_frames > 0 && frame_shift_ms > 0)
            ? static_cast<float>(original_feature_frames) * frame_shift_ms /
                  1000.0f
            : 0.0f;
    result.timestamps.reserve(generated_ids.size());
    // Linear interpolation: distribute tokens evenly over audio duration
    if (generated_ids.size() > 1 && audio_duration > 0) {
      float time_per_token =
          audio_duration / static_cast<float>(generated_ids.size());
      for (size_t i = 0; i < generated_ids.size(); ++i) {
        result.timestamps.push_back(static_cast<float>(i) * time_per_token);
      }
    } else if (generated_ids.size() == 1 && audio_duration > 0) {
      result.timestamps.push_back(audio_duration / 2.0f);
    }
  }
  return result;
}

// Decode multiple audio streams in batch.
// Applies LFR processing, runs encoder, and generates text for each stream.
void OfflineRecognizerFunASRNanoImpl::DecodeStreams(OfflineStream **ss,
                                                    int32_t n) const {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const auto &funasr_config = config_.model_config.funasr_nano;
  for (int32_t i = 0; i != n; ++i) {
    std::vector<float> f = ss[i]->GetFrames();
    f = ApplyLFR(f);
    int32_t num_frames = static_cast<int32_t>(
        f.size() / (config_.feat_config.feature_dim * model_->LfrWindowSize()));
    if (num_frames <= 0) {
      OfflineRecognitionResult r;
      r.text = "";
      ss[i]->SetResult(r);
      continue;
    }
    std::array<int64_t, 3> shape{1, num_frames,
                                  static_cast<int64_t>(f.size() / num_frames)};
    Ort::Value features = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(f.data()), f.size(), shape.data(),
        shape.size());
    Ort::Value encoder_out =
        model_->ForwardEncoderAdaptor(std::move(features));
    OfflineRecognitionResult r = GenerateText(
        std::move(encoder_out), funasr_config.system_prompt,
        funasr_config.user_prompt);
    ss[i]->SetResult(r);
  }
}

#if __ANDROID_API__ >= 9
template OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizerFunASRNanoImpl::OfflineRecognizerFunASRNanoImpl(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx
