// sherpa-onnx/csrc/axcl/offline-tts-supertonic-impl-axcl.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/csrc/axcl/offline-tts-supertonic-impl-axcl.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/axcl/offline-tts-supertonic-model-axcl.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/normal-data-generator.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {
namespace {

constexpr int32_t kFixedBatch = 1;
constexpr int32_t kFixedTextLen = 320;
constexpr int32_t kFixedLatentLen = 300;

constexpr float kMinDuration = 0.1f;

constexpr std::array<std::string_view, 31> kSupertonicAvailableLangs = {
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
    "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
    "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
};

std::string GetSupertonicAvailableLangsString() {
  std::ostringstream os;
  const char *sep = "";
  for (auto lang : kSupertonicAvailableLangs) {
    os << sep << lang;
    sep = ", ";
  }
  return os.str();
}

SupertonicStyle ParseVoiceStyleFromBinary(const std::vector<char> &buf) {
  constexpr size_t kHeaderSize = 6 * sizeof(int64_t);
  constexpr size_t kMaxPayloadBytes = 64 * 1024 * 1024;

  if (buf.size() < kHeaderSize) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style .bin: file too small (got %zu bytes, need %zu "
        "header)",
        buf.size(), kHeaderSize);
    SHERPA_ONNX_EXIT(-1);
  }
  int64_t dims[6];
  std::memcpy(dims, buf.data(), kHeaderSize);
  for (int i = 0; i < 6; ++i) {
    if (dims[i] <= 0) {
      SHERPA_ONNX_LOGE("Invalid voice style .bin: dims[%d]=%" PRId64 " <= 0", i,
                       dims[i]);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  auto mul3 = [](int64_t a, int64_t b, int64_t c, const char *name) -> size_t {
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    if (a <= 0 || b <= 0 || c <= 0 || a > kMax / b) {
      SHERPA_ONNX_LOGE("Invalid voice style .bin: %s dims overflow", name);
      SHERPA_ONNX_EXIT(-1);
    }
    int64_t ab = a * b;
    if (ab > kMax / c) {
      SHERPA_ONNX_LOGE("Invalid voice style .bin: %s dims overflow", name);
      SHERPA_ONNX_EXIT(-1);
    }
    return static_cast<size_t>(ab * c);
  };
  size_t ttl_elems = mul3(dims[0], dims[1], dims[2], "ttl");
  size_t dp_elems = mul3(dims[3], dims[4], dims[5], "dp");

  size_t ttl_bytes = ttl_elems * sizeof(float);
  size_t dp_bytes = dp_elems * sizeof(float);
  if (ttl_bytes / sizeof(float) != ttl_elems ||
      dp_bytes / sizeof(float) != dp_elems) {
    SHERPA_ONNX_LOGE("Invalid voice style .bin: byte size overflow");
    SHERPA_ONNX_EXIT(-1);
  }
  size_t payload_bytes = ttl_bytes + dp_bytes;
  if (payload_bytes < ttl_bytes || payload_bytes < dp_bytes) {
    SHERPA_ONNX_LOGE("Invalid voice style .bin: payload size overflow");
    SHERPA_ONNX_EXIT(-1);
  }
  if (payload_bytes > kMaxPayloadBytes) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style .bin: payload too large (%zu bytes, max %zu)",
        payload_bytes, kMaxPayloadBytes);
    SHERPA_ONNX_EXIT(-1);
  }
  size_t expected_total = kHeaderSize + payload_bytes;
  if (expected_total < kHeaderSize) {
    SHERPA_ONNX_LOGE("Invalid voice style .bin: total size overflow");
    SHERPA_ONNX_EXIT(-1);
  }
  if (buf.size() != expected_total) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style .bin: size mismatch (got %zu bytes, expected "
        "exactly %zu)",
        buf.size(), expected_total);
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<int64_t> ttl_shape = {dims[0], dims[1], dims[2]};
  std::vector<int64_t> dp_shape = {dims[3], dims[4], dims[5]};
  std::vector<float> ttl_data(ttl_elems);
  std::memcpy(ttl_data.data(), buf.data() + kHeaderSize, ttl_bytes);
  std::vector<float> dp_data(dp_elems);
  std::memcpy(dp_data.data(), buf.data() + kHeaderSize + ttl_bytes, dp_bytes);

  SupertonicStyle style;
  style.ttl_data = std::move(ttl_data);
  style.dp_data = std::move(dp_data);
  style.ttl_shape = std::move(ttl_shape);
  style.dp_shape = std::move(dp_shape);
  return style;
}

struct PaddedTextInputs {
  std::vector<int64_t> text_ids;  // shape [1, kFixedTextLen]
  std::vector<float> text_mask;   // shape [1, 1, kFixedTextLen]
  int64_t actual_len;
};

PaddedTextInputs PadTextInputs(const std::vector<int64_t> &text_ids_raw,
                               const std::vector<float> &text_mask_raw,
                               int64_t actual_len) {
  PaddedTextInputs out;
  out.actual_len = actual_len;
  out.text_ids.assign(kFixedBatch * kFixedTextLen, 0);
  std::copy(text_ids_raw.begin(), text_ids_raw.end(), out.text_ids.begin());
  out.text_mask.assign(kFixedBatch * 1 * kFixedTextLen, 0.0f);
  std::copy(text_mask_raw.begin(),
            text_mask_raw.begin() +
                std::min(static_cast<size_t>(actual_len), text_mask_raw.size()),
            out.text_mask.begin());
  return out;
}

struct NoisyLatentResult {
  std::vector<float> xt_flat;
  std::vector<float> latent_mask_flat;
  int32_t actual_latent_len;
};

NoisyLatentResult SampleNoisyLatentFixed(float duration, int32_t sample_rate,
                                         int32_t base_chunk_size,
                                         int32_t chunk_compress_factor,
                                         int32_t latent_dim,
                                         NormalDataGenerator &gen) {
  NoisyLatentResult out;
  int32_t wav_len = static_cast<int32_t>(duration * sample_rate);
  if (wav_len < 1) wav_len = 1;
  int32_t chunk_size = base_chunk_size * chunk_compress_factor;
  out.actual_latent_len = (wav_len + chunk_size - 1) / chunk_size;

  out.xt_flat.assign(kFixedBatch * latent_dim * kFixedLatentLen, 0.0f);
  size_t actual_noise_size = static_cast<size_t>(kFixedBatch) * latent_dim *
                             out.actual_latent_len;
  gen.Fill(out.xt_flat.data(), actual_noise_size);

  out.latent_mask_flat.assign(kFixedBatch * 1 * kFixedLatentLen, 0.0f);
  for (int i = 0; i < out.actual_latent_len; ++i) {
    out.latent_mask_flat[i] = 1.0f;
  }

  // Apply mask
  for (int b = 0; b < kFixedBatch; ++b) {
    for (int d = 0; d < latent_dim; ++d) {
      for (int t = 0; t < kFixedLatentLen; ++t) {
        size_t idx = static_cast<size_t>(b) * latent_dim * kFixedLatentLen +
                     d * kFixedLatentLen + t;
        out.xt_flat[idx] *= out.latent_mask_flat[b * kFixedLatentLen + t];
      }
    }
  }
  return out;
}

}  // namespace

OfflineTtsSupertonicImplAxcl::OfflineTtsSupertonicImplAxcl(
    const OfflineTtsConfig &config)
    : config_(config),
      model_(std::make_unique<OfflineTtsSupertonicModelAxcl>(
          config.model)),
      text_processor_(std::make_unique<SupertonicUnicodeProcessor>(
          config.model.supertonic.unicode_indexer)) {
  std::vector<char> buf = ReadFile(config.model.supertonic.voice_style);
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Failed to read voice style file: %s",
                     config.model.supertonic.voice_style.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  InitVoiceStyle(buf);
}

template <typename Manager>
OfflineTtsSupertonicImplAxcl::OfflineTtsSupertonicImplAxcl(
    Manager *mgr, const OfflineTtsConfig &config)
    : config_(config),
      model_(std::make_unique<OfflineTtsSupertonicModelAxcl>(
          mgr, config.model)),
      text_processor_(std::make_unique<SupertonicUnicodeProcessor>(
          mgr, config.model.supertonic.unicode_indexer)) {
  std::vector<char> buf = ReadFile(mgr, config.model.supertonic.voice_style);
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Failed to read voice style file: %s",
                     config.model.supertonic.voice_style.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  InitVoiceStyle(buf);
}

int32_t OfflineTtsSupertonicImplAxcl::SampleRate() const {
  return model_->GetSampleRate();
}

GeneratedAudio OfflineTtsSupertonicImplAxcl::Generate(
    const std::string &text, int64_t sid, float speed,
    GeneratedAudioCallback callback) const {
  GenerationConfig config;
  config.sid = sid;
  config.speed = speed;
  return Generate(text, config, callback);
}

GeneratedAudio OfflineTtsSupertonicImplAxcl::Generate(
    const std::string &text, const GenerationConfig &config,
    GeneratedAudioCallback callback) const {
  if (config_.model.debug) {
    SHERPA_ONNX_LOGE("%s", config.ToString().c_str());
  }
  int32_t seed = config.GetExtraInt("seed", -1);
  float speed =
      config.GetExtraFloat("speed", config.speed > 0 ? config.speed : 1.05f);
  int32_t num_steps = config.GetExtraInt(
      "num_steps", config.num_steps > 0 ? config.num_steps : 5);
  if (speed <= 0) {
    SHERPA_ONNX_LOGE("Speed must be > 0. Given: %f", speed);
    return {};
  }
  if (num_steps <= 0) {
    SHERPA_ONNX_LOGE("Num steps must be > 0. Given: %d", num_steps);
    return {};
  }
  std::string text_single = Trim(text);
  if (text_single.empty()) {
    return {};
  }

  int64_t sid = config.sid;
  if (sid >= num_speakers_ || sid < 0) {
    SHERPA_ONNX_LOGE(
        "Model has %d speaker(s). sid must be in [0, %d]. Given sid=%d, "
        "using 0",
        num_speakers_, num_speakers_ - 1, static_cast<int32_t>(sid));
    sid = 0;
  }

  std::string lang = config.GetExtraString("lang", "en");
  bool lang_ok = std::any_of(kSupertonicAvailableLangs.begin(),
                             kSupertonicAvailableLangs.end(),
                             [&](std::string_view s) { return s == lang; });
  if (!lang_ok) {
    auto available_langs = GetSupertonicAvailableLangsString();
    SHERPA_ONNX_LOGE("Invalid language: %s. Available: %s", lang.c_str(),
                     available_langs.c_str());
    return {};
  }

  float silence_duration = config.GetExtraFloat("silence_duration", 0.3f);
  size_t max_len =
      (lang == "ko" || lang == "ja")
          ? static_cast<size_t>(config.GetExtraInt("max_len", 120))
          : static_cast<size_t>(config.GetExtraInt("max_len", 300));
  if (max_len == 0) {
    SHERPA_ONNX_LOGE("Max length must be > 0. Given: %zu", max_len);
    return {};
  }
  auto text_chunks = ChunkText(text_single, max_len);
  return ProcessChunksAndConcatenate(text_chunks, lang, sid, num_steps,
                                     speed, silence_duration, seed,
                                     callback);
}

GeneratedAudio OfflineTtsSupertonicImplAxcl::Process(
    const std::string &text, const std::string &lang, int64_t sid,
    int32_t num_steps, float speed, NormalDataGenerator &gen) const {
  const auto &cfg = model_->GetConfig();
  StyleSliceView slice = GetStyleSliceForSid(sid);

  std::vector<int64_t> text_ids_raw;
  std::vector<float> text_mask_raw;
  std::vector<int64_t> text_mask_shape;
  text_processor_->Process(text, lang, &text_ids_raw, &text_mask_raw,
                           &text_mask_shape);
  if (text_ids_raw.empty() || text_mask_raw.empty()) {
    SHERPA_ONNX_LOGE(
        "Text processing failed: empty text_ids or text_mask. Text: \"%s\"",
        text.c_str());
    return {};
  }
  if (text_mask_shape.size() != 3) {
    SHERPA_ONNX_LOGE(
        "Invalid text_mask_shape size: %zu (expected 3). Text: \"%s\"",
        text_mask_shape.size(), text.c_str());
    return {};
  }
  int64_t text_seq_len = static_cast<int64_t>(text_ids_raw.size());
  int64_t text_mask_len = text_mask_shape[2];
  if (text_seq_len != text_mask_len) {
    SHERPA_ONNX_LOGE("Text sequence length mismatch: text_ids=%" PRId64
                     ", text_mask=%" PRId64 ". Text: \"%s\"",
                     text_seq_len, text_mask_len, text.c_str());
    return {};
  }

  auto padded = PadTextInputs(text_ids_raw, text_mask_raw, text_seq_len);

  // style_dp shape: [1, 8, 16]
  std::vector<float> style_dp(slice.dp_data, slice.dp_data + slice.dp_size);
  // style_ttl shape: [1, 50, 256]
  std::vector<float> style_ttl(slice.ttl_data,
                               slice.ttl_data + slice.ttl_size);

  auto dp_output = model_->RunDurationPredictor(padded.text_ids, style_dp,
                                                padded.text_mask);
  if (dp_output.size() != 1) {
    SHERPA_ONNX_LOGE(
        "Duration predictor output size mismatch: expected 1, got %zu. Text: "
        "\"%s\"",
        dp_output.size(), text.c_str());
    return {};
  }
  float duration = dp_output[0];
  duration /= speed;
  if (duration < kMinDuration) {
    duration = kMinDuration;
  }

  auto text_emb = model_->RunTextEncoder(padded.text_ids, style_ttl,
                                         padded.text_mask);
  if (text_emb.empty()) {
    SHERPA_ONNX_LOGE("Text encoder output is empty. Text: \"%s\"",
                     text.c_str());
    return {};
  }

  int32_t latent_dim = cfg.ttl.latent_dim * cfg.ttl.chunk_compress_factor;
  auto latent_result =
      SampleNoisyLatentFixed(duration, cfg.ae.sample_rate,
                             cfg.ae.base_chunk_size,
                             cfg.ttl.chunk_compress_factor, latent_dim, gen);

  std::vector<float> total_step_vec = {static_cast<float>(num_steps)};
  std::vector<float> current_step_vec = {0.0f};

  for (int32_t step = 0; step < num_steps; ++step) {
    current_step_vec[0] = static_cast<float>(step);
    auto denoised = model_->RunVectorEstimator(
        latent_result.xt_flat, current_step_vec, text_emb, style_ttl,
        latent_result.latent_mask_flat, padded.text_mask, total_step_vec);
    if (denoised.size() != latent_result.xt_flat.size()) {
      SHERPA_ONNX_LOGE(
          "Denoised latent size mismatch at step %d: expected %zu, got %zu. "
          "Text: \"%s\"",
          step, latent_result.xt_flat.size(), denoised.size(), text.c_str());
      return {};
    }
    std::memcpy(latent_result.xt_flat.data(), denoised.data(),
                denoised.size() * sizeof(float));
  }

  auto wav = model_->RunVocoder(latent_result.xt_flat);
  if (wav.empty()) {
    SHERPA_ONNX_LOGE("Vocoder output is empty. Text: \"%s\"", text.c_str());
    return {};
  }

  int64_t actual_samples =
      static_cast<int64_t>(duration * cfg.ae.sample_rate);
  if (actual_samples > static_cast<int64_t>(wav.size())) {
    actual_samples = static_cast<int64_t>(wav.size());
  }

  GeneratedAudio result;
  result.samples.assign(wav.data(), wav.data() + actual_samples);
  result.sample_rate = cfg.ae.sample_rate;

  if (config_.model.debug && !result.samples.empty()) {
    float max_abs = 0.f;
    float min_abs = std::abs(result.samples[0]);
    for (float x : result.samples) {
      float ax = std::abs(x);
      max_abs = std::max(max_abs, ax);
      min_abs = std::min(min_abs, ax);
    }
    SHERPA_ONNX_LOGE("Audio samples: %zu, min_abs=%.6f, max_abs=%.6f",
                     result.samples.size(), min_abs, max_abs);
  }
  return result;
}

GeneratedAudio
OfflineTtsSupertonicImplAxcl::ProcessChunksAndConcatenate(
    const std::vector<std::string> &text_chunks, const std::string &lang,
    int64_t sid, int32_t num_steps, float speed, float silence_duration,
    int32_t seed, GeneratedAudioCallback callback) const {
  NormalDataGenerator gen(0, 1, seed);
  GeneratedAudio result;
  std::vector<std::vector<float>> chunk_samples;
  chunk_samples.reserve(text_chunks.size());
  int32_t num_chunks = static_cast<int32_t>(text_chunks.size());
  for (int32_t i = 0; i < num_chunks; ++i) {
    auto chunk_result =
        Process(text_chunks[i], lang, sid, num_steps, speed, gen);
    if (chunk_result.samples.empty()) {
      continue;
    }
    if (callback) {
      float progress =
          static_cast<float>(i + 1) / static_cast<float>(num_chunks);
      callback(chunk_result.samples.data(), chunk_result.samples.size(),
               progress);
    }
    chunk_samples.push_back(std::move(chunk_result.samples));
  }

  if (chunk_samples.empty()) {
    result.sample_rate = model_->GetSampleRate();
    return result;
  }

  int32_t sample_rate = model_->GetSampleRate();
  size_t silence_len =
      static_cast<size_t>(silence_duration * static_cast<float>(sample_rate));
  size_t total = 0;
  for (const auto &s : chunk_samples) {
    total += s.size();
  }
  if (chunk_samples.size() > 1) {
    total += (chunk_samples.size() - 1) * silence_len;
  }

  std::vector<float> wav_cat;
  wav_cat.reserve(total);
  for (size_t i = 0; i < chunk_samples.size(); ++i) {
    if (i > 0) {
      wav_cat.insert(wav_cat.end(), silence_len, 0.f);
    }
    wav_cat.insert(wav_cat.end(), chunk_samples[i].begin(),
                   chunk_samples[i].end());
  }
  result.samples = std::move(wav_cat);
  result.sample_rate = sample_rate;
  return result;
}

void OfflineTtsSupertonicImplAxcl::InitVoiceStyle(
    const std::vector<char> &buf) {
  SupertonicStyle style = ParseVoiceStyleFromBinary(buf);
  if (style.ttl_shape.size() != 3 || style.dp_shape.size() != 3) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style: ttl_shape or dp_shape must have 3 dimensions");
    SHERPA_ONNX_EXIT(-1);
  }
  int32_t num_speakers = static_cast<int32_t>(style.ttl_shape[0]);
  if (num_speakers <= 0) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style: num_speakers must be >= 1. Given: %d",
        num_speakers);
    SHERPA_ONNX_EXIT(-1);
  }
  if (style.ttl_shape[0] != style.dp_shape[0]) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style: ttl_shape[0] != dp_shape[0]. Given: %d != %d",
        static_cast<int32_t>(style.ttl_shape[0]),
        static_cast<int32_t>(style.dp_shape[0]));
    SHERPA_ONNX_EXIT(-1);
  }
  num_speakers_ = num_speakers;
  full_style_ = std::move(style);

  if (config_.model.debug) {
    SHERPA_ONNX_LOGE("Number of speakers: %d", num_speakers_);
  }
}

OfflineTtsSupertonicImplAxcl::StyleSliceView
OfflineTtsSupertonicImplAxcl::GetStyleSliceForSid(int64_t sid) const {
  StyleSliceView out;
  int32_t s = 0;
  if (num_speakers_ != 1) {
    int64_t hi = static_cast<int64_t>(num_speakers_ - 1);
    int64_t clamped = std::clamp<int64_t>(sid, 0, hi);
    s = static_cast<int32_t>(clamped);
  }
  const SupertonicStyle &full = full_style_;
  out.ttl_shape = {1, full.ttl_shape[1], full.ttl_shape[2]};
  out.dp_shape = {1, full.dp_shape[1], full.dp_shape[2]};
  size_t ttl_slice = static_cast<size_t>(out.ttl_shape[1] * out.ttl_shape[2]);
  size_t dp_slice = static_cast<size_t>(out.dp_shape[1] * out.dp_shape[2]);
  out.ttl_size = ttl_slice;
  out.dp_size = dp_slice;
  out.ttl_data = full.ttl_data.data() + static_cast<size_t>(s) * ttl_slice;
  out.dp_data = full.dp_data.data() + static_cast<size_t>(s) * dp_slice;
  return out;
}

#if __ANDROID_API__ >= 9
template OfflineTtsSupertonicImplAxcl::OfflineTtsSupertonicImplAxcl(
    AAssetManager *mgr, const OfflineTtsConfig &config);
#endif

#if __OHOS__
template OfflineTtsSupertonicImplAxcl::OfflineTtsSupertonicImplAxcl(
    NativeResourceManager *mgr, const OfflineTtsConfig &config);
#endif

}  // namespace sherpa_onnx
