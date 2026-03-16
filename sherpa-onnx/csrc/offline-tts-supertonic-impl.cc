// sherpa-onnx/csrc/offline-tts-supertonic-impl.cc
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/csrc/offline-tts-supertonic-impl.h"

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

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/normal-data-generator.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {
namespace {

// Minimum duration (in seconds) to prevent zero-length audio
constexpr float kMinDuration = 0.1f;

// Maximum latent length to prevent excessive memory allocation and OOM.
constexpr int32_t kMaxLatentLen = 10000;

constexpr std::array<std::string_view, 5> kSupertonicAvailableLangs = {
    "en", "ko", "es", "pt", "fr",
};

void GetLatentMaskFlat(const std::vector<int64_t> &wav_lengths,
                       int32_t base_chunk_size, int32_t chunk_compress_factor,
                       std::vector<float> *mask_flat,
                       std::vector<int64_t> *mask_shape) {
  const int32_t bsz = static_cast<int32_t>(wav_lengths.size());
  int32_t wav_chunk_size = base_chunk_size * chunk_compress_factor;
  std::vector<int64_t> latent_lengths;
  latent_lengths.reserve(bsz);
  for (auto len : wav_lengths) {
    latent_lengths.push_back((len + wav_chunk_size - 1) / wav_chunk_size);
  }
  LengthsToMask(latent_lengths, mask_flat, mask_shape);
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
}  // namespace

OfflineTtsSupertonicImpl::OfflineTtsSupertonicImpl(
    const OfflineTtsConfig &config)
    : config_(config),
      model_(std::make_unique<OfflineTtsSupertonicModel>(config.model)),
      text_processor_(std::make_unique<SupertonicUnicodeProcessor>(
          config.model.supertonic.unicode_indexer)),
      memory_info_(
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
  std::vector<char> buf = ReadFile(config.model.supertonic.voice_style);
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Failed to read voice style file: %s",
                     config.model.supertonic.voice_style.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  InitVoiceStyle(buf);
}

template <typename Manager>
OfflineTtsSupertonicImpl::OfflineTtsSupertonicImpl(
    Manager *mgr, const OfflineTtsConfig &config)
    : config_(config),
      model_(std::make_unique<OfflineTtsSupertonicModel>(mgr, config.model)),
      text_processor_(std::make_unique<SupertonicUnicodeProcessor>(
          mgr, config.model.supertonic.unicode_indexer)),
      memory_info_(
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
  std::vector<char> buf = ReadFile(mgr, config.model.supertonic.voice_style);
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Failed to read voice style file: %s",
                     config.model.supertonic.voice_style.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  InitVoiceStyle(buf);
}

int32_t OfflineTtsSupertonicImpl::SampleRate() const {
  return model_->GetSampleRate();
}

GeneratedAudio OfflineTtsSupertonicImpl::Generate(
    const std::string &text, int64_t sid, float speed,
    GeneratedAudioCallback callback) const {
  GenerationConfig config;
  config.sid = sid;
  config.speed = speed;
  return Generate(text, config, callback);
}

GeneratedAudio OfflineTtsSupertonicImpl::Generate(
    const std::string &text, const GenerationConfig &config,
    GeneratedAudioCallback callback) const {
  // Supported extra options in config.extra:
  //   - "speed" (float): Speech speed factor (default: 1.05)
  //   - "num_steps" (int): Number of denoising steps (default: 5)
  //   - "lang" (string): Language code, e.g. "en", "ko" (default: "en")
  //   - sid selects speaker from voice.bin (0 .. NumSpeakers()-1).
  //   - "max_len" (int): Max chunk length. Default: 300 (non-Korean), 120 (ko).
  //   - "silence_duration" (float): Silence in seconds between chunks (default:
  //   0.3)
  //   - "seed" (int): RNG seed for reproducibility. -1 = random (default).

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
    SHERPA_ONNX_LOGE("Invalid language: %s. Available: en, ko, es, pt, fr",
                     lang.c_str());
    return {};
  }

  float silence_duration = config.GetExtraFloat("silence_duration", 0.3f);
  size_t max_len =
      (lang == "ko") ? static_cast<size_t>(config.GetExtraInt("max_len", 120))
                     : static_cast<size_t>(config.GetExtraInt("max_len", 300));
  if (max_len == 0) {
    SHERPA_ONNX_LOGE("Max length must be > 0. Given: %zu", max_len);
    return {};
  }
  auto text_chunks = ChunkText(text_single, max_len);
  return ProcessChunksAndConcatenate(text_chunks, lang, sid, num_steps, speed,
                                     silence_duration, seed, callback);
}

GeneratedAudio OfflineTtsSupertonicImpl::Process(
    const std::string &text, const std::string &lang, int64_t sid,
    int32_t num_steps, float speed, NormalDataGenerator &gen) const {
  const auto &cfg = model_->GetConfig();
  StyleSliceView slice = GetStyleSliceForSid(sid);
  const int32_t bsz = 1;

  std::vector<int64_t> text_ids;
  std::vector<float> text_mask_flat;
  std::vector<int64_t> text_mask_shape;
  text_processor_->Process(text, lang, &text_ids, &text_mask_flat,
                           &text_mask_shape);
  if (text_ids.empty() || text_mask_flat.empty()) {
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
  int64_t text_seq_len = static_cast<int64_t>(text_ids.size());
  int64_t text_mask_len = text_mask_shape[2];
  if (text_seq_len != text_mask_len) {
    SHERPA_ONNX_LOGE("Text sequence length mismatch: text_ids=%" PRId64
                     ", text_mask=%" PRId64 ". Text: \"%s\"",
                     text_seq_len, text_mask_len, text.c_str());
    return {};
  }

  std::vector<int64_t> text_ids_shape = {1, text_seq_len};

  Ort::Value text_ids_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info_, text_ids.data(), text_ids.size(), text_ids_shape.data(),
      text_ids_shape.size());
  Ort::Value style_dp_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float *>(slice.dp_data), slice.dp_size,
      slice.dp_shape.data(), slice.dp_shape.size());
  Ort::Value text_mask_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, text_mask_flat.data(), text_mask_flat.size(),
      text_mask_shape.data(), text_mask_shape.size());
  Ort::Value dp_output = model_->RunDurationPredictor(
      std::move(text_ids_tensor), std::move(style_dp_tensor),
      std::move(text_mask_tensor));
  auto dp_output_info = dp_output.GetTensorTypeAndShapeInfo();
  size_t dp_element_count = dp_output_info.GetElementCount();
  if (dp_element_count != 1) {
    SHERPA_ONNX_LOGE(
        "Duration predictor output size mismatch: expected 1, got %zu. Text: "
        "\"%s\"",
        dp_element_count, text.c_str());
    return {};
  }
  auto *dur_data = dp_output.GetTensorMutableData<float>();
  std::vector<float> duration(dur_data, dur_data + 1);
  if (speed != 1.0f) {
    for (auto &dur : duration) {
      dur /= speed;
      if (dur < kMinDuration) {
        dur = kMinDuration;
      }
    }
  }

  Ort::Value text_enc_output = model_->RunTextEncoder(
      Ort::Value::CreateTensor<int64_t>(memory_info_, text_ids.data(),
                                        text_ids.size(), text_ids_shape.data(),
                                        text_ids_shape.size()),
      Ort::Value::CreateTensor<float>(
          memory_info_, const_cast<float *>(slice.ttl_data), slice.ttl_size,
          slice.ttl_shape.data(), slice.ttl_shape.size()),
      Ort::Value::CreateTensor<float>(
          memory_info_, text_mask_flat.data(), text_mask_flat.size(),
          text_mask_shape.data(), text_mask_shape.size()));
  auto text_emb_info = text_enc_output.GetTensorTypeAndShapeInfo();
  size_t text_emb_size = text_emb_info.GetElementCount();
  if (text_emb_size == 0) {
    SHERPA_ONNX_LOGE("Text encoder output is empty. Text: \"%s\"",
                     text.c_str());
    return {};
  }
  auto *text_emb_data = text_enc_output.GetTensorMutableData<float>();
  auto text_emb_shape = text_emb_info.GetShape();

  float wav_len_max =
      *std::max_element(duration.begin(), duration.end()) * cfg.ae.sample_rate;
  std::vector<int64_t> wav_lengths;
  wav_lengths.reserve(bsz);
  for (float d : duration) {
    int64_t wav_len = static_cast<int64_t>(d * cfg.ae.sample_rate);
    if (wav_len < 1) {
      wav_len = 1;
    }
    wav_lengths.push_back(wav_len);
  }
  int32_t chunk_size = cfg.ae.base_chunk_size * cfg.ttl.chunk_compress_factor;
  int32_t latent_len =
      static_cast<int32_t>((wav_len_max + chunk_size - 1) / chunk_size);
  if (latent_len > kMaxLatentLen) {
    SHERPA_ONNX_LOGE(
        "Latent length (%d) exceeds maximum (%d), capping to prevent OOM",
        latent_len, kMaxLatentLen);
    latent_len = kMaxLatentLen;
  }

  int32_t latent_dim = cfg.ttl.latent_dim * cfg.ttl.chunk_compress_factor;
  size_t latent_total_size = static_cast<size_t>(bsz) *
                             static_cast<size_t>(latent_dim) *
                             static_cast<size_t>(latent_len);
  if (latent_total_size / static_cast<size_t>(bsz) /
          static_cast<size_t>(latent_dim) !=
      static_cast<size_t>(latent_len)) {
    SHERPA_ONNX_LOGE(
        "Latent total size overflow: bsz=%d, latent_dim=%d, latent_len=%d. "
        "Text: \"%s\"",
        bsz, latent_dim, latent_len, text.c_str());
    return {};
  }

  std::vector<float> xt_flat(latent_total_size);

  gen.Fill(xt_flat.data(), xt_flat.size());

  std::vector<float> latent_mask_flat;
  std::vector<int64_t> latent_mask_shape;
  GetLatentMaskFlat(wav_lengths, cfg.ae.base_chunk_size,
                    cfg.ttl.chunk_compress_factor, &latent_mask_flat,
                    &latent_mask_shape);
  int64_t latent_mask_len = latent_mask_shape[2];
  if (latent_mask_len != latent_len) {
    SHERPA_ONNX_LOGE("Latent mask length mismatch: expected %d, got %" PRId64
                     ". Text: \"%s\"",
                     latent_len, latent_mask_len, text.c_str());
    return {};
  }
  for (int32_t b = 0; b < bsz; ++b) {
    const float *mask_batch = latent_mask_flat.data() + b * latent_mask_len;
    float *xt_batch = xt_flat.data() + b * latent_dim * latent_len;
    for (int32_t d = 0; d < latent_dim; ++d) {
      float *xt_dim = xt_batch + d * latent_len;
      for (int32_t t = 0; t < latent_len; ++t) {
        xt_dim[t] *= mask_batch[t];
      }
    }
  }

  std::vector<int64_t> latent_shape = {bsz, latent_dim, latent_len};
  std::vector<float> total_step_vec(bsz, static_cast<float>(num_steps));
  std::array<int64_t, 1> step_shape = {bsz};

  // Constant inputs: create once outside loop, keep text_enc_output alive.
  Ort::Value text_emb_const = Ort::Value::CreateTensor<float>(
      memory_info_, text_emb_data, text_emb_size, text_emb_shape.data(),
      text_emb_shape.size());
  Ort::Value style_ttl_const = Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float *>(slice.ttl_data), slice.ttl_size,
      slice.ttl_shape.data(), slice.ttl_shape.size());
  Ort::Value text_mask_const = Ort::Value::CreateTensor<float>(
      memory_info_, text_mask_flat.data(), text_mask_flat.size(),
      text_mask_shape.data(), text_mask_shape.size());
  Ort::Value latent_mask_const = Ort::Value::CreateTensor<float>(
      memory_info_, latent_mask_flat.data(), latent_mask_flat.size(),
      latent_mask_shape.data(), latent_mask_shape.size());
  Ort::Value total_step_const = Ort::Value::CreateTensor<float>(
      memory_info_, total_step_vec.data(), total_step_vec.size(),
      step_shape.data(), step_shape.size());

  float current_step = 0.f;
  for (int32_t step = 0; step < num_steps; step++) {
    current_step = static_cast<float>(step);
    Ort::Value noisy_latent_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, xt_flat.data(), xt_flat.size(), latent_shape.data(),
        latent_shape.size());
    Ort::Value current_step_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, &current_step, 1, step_shape.data(), step_shape.size());

    Ort::Value vector_est_output = model_->RunVectorEstimator(
        std::move(noisy_latent_tensor), std::move(current_step_tensor),
        text_emb_const, style_ttl_const, latent_mask_const, text_mask_const,
        total_step_const);
    auto vector_est_output_info = vector_est_output.GetTensorTypeAndShapeInfo();
    size_t denoised_size = vector_est_output_info.GetElementCount();
    if (denoised_size != latent_total_size) {
      SHERPA_ONNX_LOGE(
          "Denoised latent size mismatch at step %d: expected %zu, got %zu. "
          "Text: \"%s\"",
          step, latent_total_size, denoised_size, text.c_str());
      return {};
    }
    auto *denoised_data = vector_est_output.GetTensorMutableData<float>();
    std::memcpy(xt_flat.data(), denoised_data,
                latent_total_size * sizeof(float));
  }

  Ort::Value latent_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, xt_flat.data(), xt_flat.size(), latent_shape.data(),
      latent_shape.size());
  Ort::Value vocoder_output = model_->RunVocoder(std::move(latent_tensor));
  auto wav_info = vocoder_output.GetTensorTypeAndShapeInfo();
  auto wav_shape = wav_info.GetShape();
  size_t wav_size = wav_info.GetElementCount();
  if (wav_size == 0) {
    SHERPA_ONNX_LOGE("Vocoder output is empty. Text: \"%s\"", text.c_str());
    return {};
  }

  auto *wav_data = vocoder_output.GetTensorMutableData<float>();
  if (config_.model.debug) {
    std::ostringstream os;
    os << "Vocoder output shape: [";
    for (size_t i = 0; i < wav_shape.size(); ++i) {
      if (i > 0) os << ", ";
      os << wav_shape[i];
    }
    os << "], total elements: " << wav_size << ", bsz: " << bsz;
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }

  GeneratedAudio result;
  if ((wav_shape.size() == 2 && wav_shape[0] == bsz) ||
      (wav_shape.size() == 3 && wav_shape[0] == bsz && wav_shape[1] == 1)) {
    int64_t samples_per_batch =
        (wav_shape.size() == 2) ? wav_shape[1] : wav_shape[2];
    result.samples.reserve(static_cast<size_t>(std::accumulate(
        wav_lengths.begin(), wav_lengths.end(), static_cast<int64_t>(0))));
    for (int32_t b = 0; b < bsz; ++b) {
      int64_t actual_len = wav_lengths[b];
      if (actual_len > samples_per_batch) {
        actual_len = samples_per_batch;
      }
      const float *batch_wav = wav_data + b * samples_per_batch;
      result.samples.insert(result.samples.end(), batch_wav,
                            batch_wav + actual_len);
    }
  } else if (wav_shape.size() == 1 ||
             (wav_shape.size() == 2 && wav_shape[0] == 1)) {
    result.samples.assign(wav_data, wav_data + wav_size);
  } else {
    std::ostringstream os;
    os << "Unexpected vocoder output shape: [";
    for (size_t i = 0; i < wav_shape.size(); ++i) {
      if (i > 0) os << ", ";
      os << wav_shape[i];
    }
    os << "], bsz=" << bsz << ", using all samples";
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
    result.samples.assign(wav_data, wav_data + wav_size);
  }
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
  result.sample_rate = cfg.ae.sample_rate;
  return result;
}

GeneratedAudio OfflineTtsSupertonicImpl::ProcessChunksAndConcatenate(
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

void OfflineTtsSupertonicImpl::InitVoiceStyle(const std::vector<char> &buf) {
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

OfflineTtsSupertonicImpl::StyleSliceView
OfflineTtsSupertonicImpl::GetStyleSliceForSid(int64_t sid) const {
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
template OfflineTtsSupertonicImpl::OfflineTtsSupertonicImpl(
    AAssetManager *mgr, const OfflineTtsConfig &config);
#endif

#if __OHOS__
template OfflineTtsSupertonicImpl::OfflineTtsSupertonicImpl(
    NativeResourceManager *mgr, const OfflineTtsConfig &config);
#endif

}  // namespace sherpa_onnx
