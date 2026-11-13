// sherpa-onnx/csrc/offline-recognizer-audio8-asr-impl.cc
//
// Copyright (c)  2026  yuyun2000

#include "sherpa-onnx/csrc/offline-recognizer-audio8-asr-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
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

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-audio8-asr-model.h"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

namespace {

// Audio8-ASR uses the Whisper feature extractor: 128 mel bins, 10 ms hop.
constexpr int32_t kAudio8MelDim = 128;

// The encoder graph is exported for a full 30 s window (3000 frames).
constexpr int32_t kAudio8MaxFrames = 3000;

// Conv frontend: 100 mel frames per chunk -> 13 encoder tokens.
constexpr int32_t kAudio8ChunkFrames = 100;
constexpr int32_t kAudio8TokensPerChunk = 13;

// Three stride-2 convolutions.
int32_t AfterCnn(int32_t x) {
  if (x <= 0) {
    return 0;
  }
  x = (x - 1) / 2 + 1;
  x = (x - 1) / 2 + 1;
  return (x - 1) / 2 + 1;
}

// Number of encoder rows that belong to `num_mel_frames` of real audio.
int32_t Audio8EncoderRows(int32_t num_mel_frames) {
  if (num_mel_frames <= 0) {
    return 0;
  }
  int32_t full = num_mel_frames / kAudio8ChunkFrames;
  int32_t rem = num_mel_frames % kAudio8ChunkFrames;
  return full * kAudio8TokensPerChunk + (rem > 0 ? AfterCnn(rem) : 0);
}

// Number of `<|audio|>` tokens the prompt uses: ((mel+1)/2)/4.
int32_t Audio8AudioTokenCount(int32_t num_mel_frames) {
  int32_t downsampled = (num_mel_frames + 1) / 2;
  return std::max(1, downsampled / 4);
}

// F.adaptive_avg_pool1d over the row dimension.
void Audio8AdaptiveAvgPool(const float *in, int32_t num_in_rows, int32_t dim,
                           int32_t num_out_rows, float *out) {
  for (int32_t i = 0; i != num_out_rows; ++i) {
    int32_t start = static_cast<int32_t>(
        std::floor(static_cast<double>(i) * num_in_rows / num_out_rows));
    int32_t end = static_cast<int32_t>(
        std::ceil(static_cast<double>(i + 1) * num_in_rows / num_out_rows));
    start = std::max(0, std::min(start, num_in_rows - 1));
    end = std::max(start + 1, std::min(end, num_in_rows));
    float *dst = out + static_cast<size_t>(i) * dim;
    std::fill(dst, dst + dim, 0.0f);
    for (int32_t r = start; r != end; ++r) {
      const float *src = in + static_cast<size_t>(r) * dim;
      for (int32_t d = 0; d != dim; ++d) {
        dst[d] += src[d];
      }
    }
    const float scale = 1.0f / static_cast<float>(end - start);
    for (int32_t d = 0; d != dim; ++d) {
      dst[d] *= scale;
    }
  }
}

inline bool IsBlockToken(int32_t id) { return id >= 151646 && id <= 151650; }

bool IsDegenerateRepetition(const std::vector<int64_t> &ids) {
  const size_t window = 12;
  if (ids.size() < window) {
    return false;
  }
  for (size_t i = ids.size() - window + 1; i < ids.size(); ++i) {
    if (ids[i] != ids[i - 1]) {
      return false;
    }
  }
  return true;
}

void TrimDegenerateTail(std::vector<int64_t> *ids) {
  while (ids->size() > 1 && ids->back() == ids->at(ids->size() - 2)) {
    ids->pop_back();
  }
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

OfflineRecognizerAudio8AsrImpl::OfflineRecognizerAudio8AsrImpl(
    const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(config),
      config_(config),
      model_(std::make_unique<OfflineAudio8AsrModel>(config.model_config)),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          config.model_config.audio8_asr.tokenizer)) {
  InitPromptIds();
}

template <typename Manager>
OfflineRecognizerAudio8AsrImpl::OfflineRecognizerAudio8AsrImpl(
    Manager *mgr, const OfflineRecognizerConfig &config)
    : OfflineRecognizerImpl(mgr, config),
      config_(config),
      model_(std::make_unique<OfflineAudio8AsrModel>(mgr, config.model_config)),
      tokenizer_(std::make_unique<QwenAsrTokenizer>(
          mgr, config.model_config.audio8_asr.tokenizer)) {
  InitPromptIds();
}

std::unique_ptr<OfflineStream> OfflineRecognizerAudio8AsrImpl::CreateStream()
    const {
  WhisperTag tag;
  tag.dim = kAudio8MelDim;
  // Same centered STFT convention as the Whisper feature extractor.
  tag.align_to_stft_center = true;
  return std::make_unique<OfflineStream>(tag);
}

void OfflineRecognizerAudio8AsrImpl::InitPromptIds() {
  prompt_prefix_ids_ = tokenizer_->Encode(std::string("<|user|>") +
                                          "<|begin_of_audio|>");
  prompt_suffix_ids_ = tokenizer_->Encode(
      "<|end_of_audio|>Please transcribe this audio.<|assistant|>");

  auto audio_ids = tokenizer_->Encode("<|audio|>");
  if (audio_ids.size() != 1) {
    SHERPA_ONNX_LOGE(
        "Failed to tokenize <|audio|> into a single token for audio8-asr");
    SHERPA_ONNX_EXIT(-1);
  }
  audio_token_id_ = static_cast<int32_t>(audio_ids[0]);

  eos_id_ = static_cast<int32_t>(tokenizer_->GetImEndTokenId());
  pad_id_ = static_cast<int32_t>(tokenizer_->GetPadTokenId());
}

int64_t OfflineRecognizerAudio8AsrImpl::SampleToken(const float *logits,
                                                    int32_t vocab_size) const {
  int32_t best = 0;
  float best_val = -std::numeric_limits<float>::infinity();
  for (int32_t i = 0; i != vocab_size; ++i) {
    float v = logits[i];
    if (std::isfinite(v) && v > best_val) {
      best_val = v;
      best = i;
    }
  }
  return best;
}

OfflineRecognitionResult OfflineRecognizerAudio8AsrImpl::GenerateText(
    std::vector<float> *audio_features, int32_t num_audio_tokens,
    int32_t audio_dim) const {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  OrtAllocator *allocator = model_->Allocator();

  const auto &c = config_.model_config.audio8_asr;
  const int32_t model_max_len = model_->GetMaxTotalLen();
  const int32_t max_total_len =
      std::min(model_max_len, c.max_total_len > 0 ? c.max_total_len
                                                 : model_max_len);

  // ---- prompt -------------------------------------------------------------
  std::vector<int64_t> ids;
  ids.reserve(prompt_prefix_ids_.size() + num_audio_tokens +
              prompt_suffix_ids_.size());
  ids.insert(ids.end(), prompt_prefix_ids_.begin(), prompt_prefix_ids_.end());
  for (int32_t i = 0; i != num_audio_tokens; ++i) {
    ids.push_back(audio_token_id_);
  }
  ids.insert(ids.end(), prompt_suffix_ids_.begin(), prompt_suffix_ids_.end());

  const int32_t prompt_len = static_cast<int32_t>(ids.size());
  const int32_t max_new_tokens = std::min(
      c.max_new_tokens, max_total_len - prompt_len - 1);
  if (max_new_tokens <= 0) {
    SHERPA_ONNX_LOGE(
        "audio8-asr: prompt (%d) + 1 exceeds max_total_len (%d); shorten the "
        "audio or export a decoder with a larger cache",
        prompt_len, max_total_len);
    return {};
  }

  // ---- KV cache -----------------------------------------------------------
  auto cache = model_->CreateEmptyKVCache(1);

  // ---- prefill ------------------------------------------------------------
  std::array<int64_t, 2> ids_shape{1, static_cast<int64_t>(prompt_len)};
  Ort::Value input_ids = Ort::Value::CreateTensor<int64_t>(
      memory_info, ids.data(), ids.size(), ids_shape.data(), ids_shape.size());

  std::array<int64_t, 3> audio_shape{1, static_cast<int64_t>(num_audio_tokens),
                                     audio_dim};
  if (audio_features->size() !=
      static_cast<size_t>(num_audio_tokens) * audio_dim) {
    SHERPA_ONNX_LOGE("audio8-asr: audio feature size mismatch");
    return {};
  }
  Ort::Value audio_hidden = Ort::Value::CreateTensor<float>(
      memory_info, audio_features->data(), audio_features->size(),
      audio_shape.data(), audio_shape.size());

  std::vector<int64_t> attn_mask(static_cast<size_t>(max_total_len), 0);
  std::fill_n(attn_mask.begin(), prompt_len, 1);
  std::array<int64_t, 2> mask_shape{1, max_total_len};
  Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(
      memory_info, attn_mask.data(), attn_mask.size(), mask_shape.data(),
      mask_shape.size());

  std::vector<int64_t> cache_position(static_cast<size_t>(prompt_len));
  for (int32_t i = 0; i != prompt_len; ++i) {
    cache_position[i] = i;
  }
  std::array<int64_t, 1> pos_shape{static_cast<int64_t>(prompt_len)};
  Ort::Value pos = Ort::Value::CreateTensor<int64_t>(
      memory_info, cache_position.data(), cache_position.size(),
      pos_shape.data(), pos_shape.size());

  auto [logits, delta] =
      model_->ForwardLM(std::move(input_ids), std::move(audio_hidden),
                        std::move(attention_mask), pos, cache);
  model_->ApplyKvDeltaInplace(&cache, delta, pos);

  if (config_.model_config.debug) {
    std::string head, tail;
    for (int32_t i = 0; i != std::min<int32_t>(8, prompt_len); ++i) {
      head += std::to_string(ids[i]) + " ";
    }
    for (int32_t i = std::max(0, prompt_len - 8); i != prompt_len; ++i) {
      tail += std::to_string(ids[i]) + " ";
    }
    SHERPA_ONNX_LOGE(
        "audio8-asr: prompt_len=%d head=[%s] tail=[%s] eos=%d pad=%d",
        prompt_len, head.c_str(), tail.c_str(), eos_id_, pad_id_);
  }

  auto logits_info = logits.GetTensorTypeAndShapeInfo();
  auto logits_shape = logits_info.GetShape();
  const int32_t vocab_size = static_cast<int32_t>(logits_shape.back());
  const float *logits_data = logits.GetTensorData<float>();
  const float *last =
      logits_data + static_cast<size_t>(prompt_len - 1) * vocab_size;

  // ---- greedy decode ------------------------------------------------------
  std::vector<int64_t> generated;
  generated.reserve(max_new_tokens);
  std::vector<float> last_logits(last, last + vocab_size);

  int32_t position = prompt_len;
  for (int32_t step = 0; step != max_new_tokens; ++step) {
    const int64_t next = SampleToken(last_logits.data(), vocab_size);
    if (static_cast<int32_t>(next) == eos_id_ ||
        static_cast<int32_t>(next) == pad_id_) {
      break;
    }

    // Block ids are prompt scaffolding: keep them out of the transcript, but
    // still feed them back to the decoder so that the state keeps advancing.
    const bool emit = !IsBlockToken(static_cast<int32_t>(next));
    if (emit) {
      generated.push_back(next);
      if (IsDegenerateRepetition(generated)) {
        SHERPA_ONNX_LOGE(
            "audio8-asr: decode collapsed into a repetition loop after %d "
            "tokens; truncating the repetition",
            static_cast<int32_t>(generated.size()));
        TrimDegenerateTail(&generated);
        break;
      }
    }

    std::array<int64_t, 2> step_ids_shape{1, 1};
    int64_t step_id = next;
    Ort::Value step_ids = Ort::Value::CreateTensor<int64_t>(
        memory_info, &step_id, 1, step_ids_shape.data(),
        step_ids_shape.size());

    std::fill(attn_mask.begin(), attn_mask.end(), 0);
    std::fill_n(attn_mask.begin(), position + 1, 1);
    Ort::Value step_mask = Ort::Value::CreateTensor<int64_t>(
        memory_info, attn_mask.data(), attn_mask.size(), mask_shape.data(),
        mask_shape.size());

    std::array<int64_t, 1> step_pos_shape{1};
    int64_t step_pos = position;
    Ort::Value step_pos_t = Ort::Value::CreateTensor<int64_t>(
        memory_info, &step_pos, 1, step_pos_shape.data(),
        step_pos_shape.size());

    std::array<int64_t, 3> step_audio_shape{
        1, static_cast<int64_t>(num_audio_tokens), audio_dim};
    Ort::Value step_audio = Ort::Value::CreateTensor<float>(
        memory_info, audio_features->data(), audio_features->size(),
        step_audio_shape.data(), step_audio_shape.size());

    auto [step_logits, step_delta] =
        model_->ForwardLM(std::move(step_ids), std::move(step_audio),
                          std::move(step_mask), step_pos_t, cache);
    model_->ApplyKvDeltaInplace(&cache, step_delta, step_pos_t);

    const float *row = step_logits.GetTensorData<float>();
    last_logits.assign(row, row + vocab_size);
    ++position;
  }

  OfflineRecognitionResult result;
  result.text = tokenizer_->Decode(generated);
  RemoveUtf8ReplacementChars(&result.text);
  result.text = ApplyInverseTextNormalization(std::move(result.text));
  result.tokens.reserve(generated.size());
  std::string pending;
  for (int64_t id : generated) {
    result.tokens.push_back(tokenizer_->GetTokenStringStreaming(id, &pending));
  }
  return result;
}

void OfflineRecognizerAudio8AsrImpl::Decode(OfflineStream *stream) const {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::vector<float> f = stream->GetFrames();
  if (f.empty()) {
    OfflineRecognitionResult r;
    stream->SetResult(r);
    return;
  }

  int32_t num_frames = static_cast<int32_t>(f.size() / kAudio8MelDim);
  if (static_cast<size_t>(num_frames) * kAudio8MelDim != f.size() ||
      num_frames < 2) {
    OfflineRecognitionResult r;
    stream->SetResult(r);
    return;
  }

  num_frames = std::min(num_frames, kAudio8MaxFrames);
  NormalizeWhisperFeatures(f.data(), num_frames, kAudio8MelDim);

  // ---- pad the mel to the 30 s window expected by the encoder ---------------
  std::vector<float> mel(
      static_cast<size_t>(kAudio8MelDim) * kAudio8MaxFrames, 0.0f);
  for (int32_t t = 0; t != num_frames; ++t) {
    for (int32_t c = 0; c != kAudio8MelDim; ++c) {
      mel[static_cast<size_t>(c) * kAudio8MaxFrames + t] =
          f[static_cast<size_t>(t) * kAudio8MelDim + c];
    }
  }

  std::array<int64_t, 3> mel_shape{1, kAudio8MelDim, kAudio8MaxFrames};
  Ort::Value audios = Ort::Value::CreateTensor<float>(
      memory_info, mel.data(), mel.size(), mel_shape.data(), mel_shape.size());

  std::vector<int64_t> rows(1, Audio8EncoderRows(num_frames));
  std::array<int64_t, 1> rows_shape{1};
  Ort::Value num_rows = Ort::Value::CreateTensor<int64_t>(
      memory_info, rows.data(), 1, rows_shape.data(), rows_shape.size());

  Ort::Value hidden =
      model_->ForwardEncoder(std::move(audios), std::move(num_rows));
  auto hidden_shape = hidden.GetTensorTypeAndShapeInfo().GetShape();
  const int32_t hidden_dim = static_cast<int32_t>(hidden_shape.back());
  const int32_t num_rows_valid =
      std::min(static_cast<int32_t>(rows[0]),
               static_cast<int32_t>(hidden_shape[1]));
  if (num_rows_valid <= 0) {
    OfflineRecognitionResult r;
    stream->SetResult(r);
    return;
  }

  const int32_t num_audio_tokens = Audio8AudioTokenCount(num_frames);
  std::vector<float> audio_features(
      static_cast<size_t>(num_audio_tokens) * hidden_dim);
  Audio8AdaptiveAvgPool(hidden.GetTensorData<float>(), num_rows_valid,
                        hidden_dim, num_audio_tokens,
                        audio_features.data());

  if (config_.model_config.debug) {
    SHERPA_ONNX_LOGE(
        "audio8-asr: mel_frames=%d encoder_rows=%d audio_tokens=%d", num_frames,
        num_rows_valid, num_audio_tokens);
  }

  OfflineRecognitionResult r =
      GenerateText(&audio_features, num_audio_tokens, hidden_dim);
  stream->SetResult(r);
}

void OfflineRecognizerAudio8AsrImpl::DecodeStreams(OfflineStream **ss,
                                                   int32_t n) const {
  for (int32_t i = 0; i != n; ++i) {
    Decode(ss[i]);
  }
}

#if __ANDROID_API__ >= 9
template OfflineRecognizerAudio8AsrImpl::OfflineRecognizerAudio8AsrImpl(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizerAudio8AsrImpl::OfflineRecognizerAudio8AsrImpl(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_onnx
