// sherpa-onnx/csrc/offline-tts-supertonic-impl.cc
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/csrc/offline-tts-supertonic-impl.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>
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
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {
namespace {

// Minimum duration (in seconds) to prevent zero-length audio
constexpr float kMinDuration = 0.1f;

// Maximum latent length to prevent excessive memory allocation and OOM.
constexpr int kMaxLatentLen = 10000;

// Compute product of dims with overflow check. Exits on overflow.
static int64_t ComputeDimsProduct(const std::vector<int64_t> &dims,
                                  const char *name_for_error) {
  int64_t product = 1;
  for (int64_t d : dims) {
    if (d <= 0) {
      SHERPA_ONNX_LOGE("Invalid voice style: %s dim %ld <= 0", name_for_error,
                       d);
      SHERPA_ONNX_EXIT(-1);
    }
    if (product > INT64_MAX / d) {
      SHERPA_ONNX_LOGE("Invalid voice style: %s product overflow (dim=%ld)",
                       name_for_error, d);
      SHERPA_ONNX_EXIT(-1);
    }
    product *= d;
  }
  return product;
}

static void GetLatentMaskFlat(const std::vector<int64_t> &wav_lengths, int bsz,
                              int base_chunk_size, int chunk_compress_factor,
                              int64_t latent_len, std::vector<float> *mask_flat,
                              std::vector<int64_t> *mask_shape) {
  int latent_size = base_chunk_size * chunk_compress_factor;
  std::vector<int64_t> latent_lengths;
  latent_lengths.reserve(bsz);
  for (auto len : wav_lengths) {
    latent_lengths.push_back((len + latent_size - 1) / latent_size);
  }
  LengthToMaskFlat(latent_lengths, bsz, latent_len, mask_flat, mask_shape);
}

static std::vector<std::string> ChunkText(const std::string &text,
                                          size_t max_len) {
  std::vector<std::string> chunks;
  std::regex paragraph_regex(R"(\n\s*\n+)");
  std::sregex_token_iterator iter(text.begin(), text.end(), paragraph_regex,
                                  -1);
  std::sregex_token_iterator end;
  std::vector<std::string> paragraphs;
  for (; iter != end; ++iter) {
    std::string para = Trim(*iter);
    if (!para.empty()) {
      paragraphs.push_back(para);
    }
  }
  // Split by sentence delimiters but keep the punctuation
  std::regex sentence_regex(R"(([.!?])(\s+|$))");
  for (const auto &paragraph : paragraphs) {
    std::vector<std::string> sentences;
    std::sregex_iterator match_iter(paragraph.begin(), paragraph.end(),
                                    sentence_regex);
    size_t last_pos = 0;
    for (; match_iter != std::sregex_iterator(); ++match_iter) {
      size_t match_pos = match_iter->position(0);
      std::string sentence = paragraph.substr(last_pos, match_pos - last_pos);
      // Include the punctuation (match_iter->str(1) is the punctuation)
      sentence += match_iter->str(1);
      if (!sentence.empty()) {
        sentences.push_back(sentence);
      }
      last_pos = match_iter->position(0) + match_iter->length(0);
    }
    // Add remaining text after last sentence delimiter
    if (last_pos < paragraph.size()) {
      std::string remaining = paragraph.substr(last_pos);
      if (!remaining.empty()) {
        sentences.push_back(remaining);
      }
    }
    std::string current_chunk;
    for (const auto &sentence : sentences) {
      size_t total_len = current_chunk.length() + sentence.length();
      // Add space separator length if chunk is not empty
      if (!current_chunk.empty()) {
        total_len += 1;
      }
      if (total_len <= max_len) {
        if (!current_chunk.empty()) {
          current_chunk += " ";
        }
        current_chunk += sentence;
      } else {
        if (!current_chunk.empty()) {
          chunks.push_back(Trim(current_chunk));
        }
        current_chunk = sentence;
      }
    }
    if (!current_chunk.empty()) {
      chunks.push_back(Trim(current_chunk));
    }
  }
  if (chunks.empty()) {
    chunks.push_back(Trim(text));
  }
  return chunks;
}

static std::string GetVoicePath(const std::string &voice_style) {
  std::string path = Trim(voice_style);
  if (path.empty()) {
    SHERPA_ONNX_LOGE("No voice style path in config");
    SHERPA_ONNX_EXIT(-1);
  }
  return path;
}

static SupertonicStyle ParseVoiceStyleFromBinary(const std::vector<char> &buf) {
  constexpr size_t kHeaderSize = 6 * sizeof(int64_t);
  if (buf.size() < kHeaderSize) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style .bin: file too small (got %zu bytes, need %zu "
        "header)",
        buf.size(), kHeaderSize);
    SHERPA_ONNX_EXIT(-1);
  }

  int64_t dims[6];
  std::memcpy(dims, buf.data(), kHeaderSizer);
  std::vector<int64_t> ttl_shape = {dims[0], dims[1], dims[2]};
  std::vector<int64_t> dp_shape = {dims[3], dims[4], dims[5]};
  int64_t ttl_expected_size = ComputeDimsProduct(ttl_shape, "ttl_shape");
  int64_t dp_expected_size = ComputeDimsProduct(dp_shape, "dp_shape");
  size_t ttl_data_size_bytes =
      static_cast<size_t>(ttl_expected_size) * sizeof(float);
  size_t dp_data_size_bytes =
      static_cast<size_t>(dp_expected_size) * sizeof(float);
  size_t expected_total_size =
      kHeaderSize + ttl_data_size_bytes + dp_data_size_bytes;
  if (buf.size() < expected_total_size) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style .bin: file too small (got %zu bytes, expected "
        "%zu)",
        buf.size(), expected_total_size);
    SHERPA_ONNX_EXIT(-1);
  }
  std::vector<float> ttl_data(static_cast<size_t>(ttl_expected_size));
  std::vector<float> dp_data(static_cast<size_t>(dp_expected_size));
  std::memcpy(ttl_data.data(), buf.data() + kHeaderSize, ttl_data_size_bytes);
  std::memcpy(dp_data.data(), buf.data() + kHeaderSize + ttl_data_size_bytes,
              dp_data_size_bytes);
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
          ResolveAbsolutePath(config.model.supertonic.model_dir) +
          "/unicode_indexer.json")) {
  std::string voice_path = GetVoicePath(config.model.supertonic.voice_style);
  std::vector<char> buf = ReadFile(ResolveAbsolutePath(voice_path));
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Failed to read voice style file: %s", voice_path.c_str());
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
          mgr, config.model.supertonic.model_dir + "/unicode_indexer.json")) {
  std::string voice_path = GetVoicePath(config.model.supertonic.voice_style);
  std::vector<char> buf = ReadFile(mgr, voice_path);
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Failed to read voice style file: %s", voice_path.c_str());
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
  //   - "max_len_korean" (int): Max chunk length for Korean (default: 120)
  //   - "max_len_other" (int): Max chunk length for other languages (default:
  //   300)
  float speed =
      config.GetExtraFloat("speed", config.speed > 0 ? config.speed : 1.05f);
  int32_t num_steps = config.GetExtraInt(
      "num_steps", config.num_steps > 0 ? config.num_steps : 5);
  if (speed <= 0) {
    SHERPA_ONNX_LOGE("Speed must be > 0. Given: %f", speed);
    SHERPA_ONNX_EXIT(-1);
  }
  if (num_steps <= 0) {
    SHERPA_ONNX_LOGE("Num steps must be > 0. Given: %d", num_steps);
    SHERPA_ONNX_EXIT(-1);
  }
  std::string text_single = Trim(text);
  if (text_single.empty()) {
    SHERPA_ONNX_LOGE("Input text is empty");
    SHERPA_ONNX_EXIT(-1);
  }

  int64_t sid = config.sid;
  if (num_speakers_ > 0 && (sid >= num_speakers_ || sid < 0)) {
    SHERPA_ONNX_LOGE(
        "Model has %d speaker(s). sid must be in [0, %d]. Given sid=%d, "
        "using 0",
        num_speakers_, num_speakers_ - 1, static_cast<int32_t>(sid));
    sid = 0;
  }
  SupertonicStyle style = GetStyleForSid(sid);

  std::string lang_str = config.GetExtraString("lang", "");
  std::string lang =
      lang_str.empty() ? "en" : Trim(lang_str.substr(0, lang_str.find(',')));
  if (lang.empty()) {
    lang = "en";
  }
  float silence_duration = 0.3f;
  size_t max_len_cfg =
      (lang == "ko")
          ? static_cast<size_t>(config.GetExtraInt("max_len_korean", 120))
          : static_cast<size_t>(config.GetExtraInt("max_len_other", 300));
  if (max_len_cfg <= 0) {
    SHERPA_ONNX_LOGE("Max length must be > 0. Given: %d", max_len_cfg);
    SHERPA_ONNX_EXIT(-1);
  }
  size_t max_len = static_cast<size_t>(max_len_cfg);
  auto text_chunks = ChunkText(text_single, max_len);
  return ProcessChunksAndConcatenate(text_chunks, lang, style, num_steps, speed,
                                     silence_duration, callback);
}

GeneratedAudio OfflineTtsSupertonicImpl::Process(
    const std::vector<std::string> &text_list,
    const std::vector<std::string> &lang_list, const SupertonicStyle &style,
    int32_t num_steps, float speed) const {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const auto &cfg = model_->GetConfig();
  int32_t bsz = static_cast<int32_t>(text_list.size());
  if (bsz != static_cast<int32_t>(style.ttl_shape[0])) {
    SHERPA_ONNX_LOGE(
        "Number of texts (%d) must match number of style vectors (%d). "
        "When using a multi-speaker voice.bin, pass style from "
        "GetStyleForSid(sid) with one row per text.",
        bsz, static_cast<int32_t>(style.ttl_shape[0]));
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<std::vector<int64_t>> text_ids;
  std::vector<float> text_mask_flat;
  std::vector<int64_t> text_mask_shape;
  text_processor_->Process(text_list, lang_list, &text_ids, &text_mask_flat,
                           &text_mask_shape);
  if (text_ids.empty() || text_mask_flat.empty()) {
    SHERPA_ONNX_LOGE("Text processing failed: empty text_ids or text_mask");
    SHERPA_ONNX_EXIT(-1);
  }
  if (text_mask_shape.size() != 3) {
    SHERPA_ONNX_LOGE("Invalid text_mask_shape size: %zu (expected 3)",
                     text_mask_shape.size());
    SHERPA_ONNX_EXIT(-1);
  }
  int64_t text_seq_len = static_cast<int64_t>(text_ids[0].size());
  int64_t text_mask_len = text_mask_shape[2];
  if (text_seq_len != text_mask_len) {
    SHERPA_ONNX_LOGE("Text sequence length mismatch: text_ids=%d, text_mask=%d",
                     static_cast<int32_t>(text_seq_len),
                     static_cast<int32_t>(text_mask_len));
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<int64_t> text_ids_shape = {bsz, text_seq_len};
  std::vector<int64_t> text_ids_flat;
  text_ids_flat.reserve(bsz * text_seq_len);
  for (const auto &row : text_ids) {
    text_ids_flat.insert(text_ids_flat.end(), row.begin(), row.end());
  }

  Ort::Value style_dp_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(style.dp_data.data()),
      style.dp_data.size(), style.dp_shape.data(), style.dp_shape.size());
  std::vector<Ort::Value> dp_inputs;
  dp_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
      memory_info, text_ids_flat.data(), text_ids_flat.size(),
      text_ids_shape.data(), text_ids_shape.size()));
  dp_inputs.push_back(std::move(style_dp_tensor));
  dp_inputs.push_back(Ort::Value::CreateTensor<float>(
      memory_info, text_mask_flat.data(), text_mask_flat.size(),
      text_mask_shape.data(), text_mask_shape.size()));
  Ort::Value dp_output = model_->RunDurationPredictor(std::move(dp_inputs));
  auto dp_output_info = dp_output.GetTensorTypeAndShapeInfo();
  size_t dp_element_count = dp_output_info.GetElementCount();
  if (dp_element_count != static_cast<size_t>(bsz)) {
    SHERPA_ONNX_LOGE(
        "Duration predictor output size mismatch: expected %d, got %zu", bsz,
        dp_element_count);
    SHERPA_ONNX_EXIT(-1);
  }
  auto *dur_data = dp_output.GetTensorMutableData<float>();
  std::vector<float> duration(dur_data, dur_data + bsz);
  for (auto &dur : duration) {
    dur /= speed;
    if (dur < kMinDuration) {
      dur = kMinDuration;
    }
  }

  std::vector<Ort::Value> text_enc_inputs;
  text_enc_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
      memory_info, text_ids_flat.data(), text_ids_flat.size(),
      text_ids_shape.data(), text_ids_shape.size()));
  text_enc_inputs.push_back(Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(style.ttl_data.data()),
      style.ttl_data.size(), style.ttl_shape.data(), style.ttl_shape.size()));
  text_enc_inputs.push_back(Ort::Value::CreateTensor<float>(
      memory_info, text_mask_flat.data(), text_mask_flat.size(),
      text_mask_shape.data(), text_mask_shape.size()));
  Ort::Value text_enc_output =
      model_->RunTextEncoder(std::move(text_enc_inputs));
  auto text_emb_info = text_enc_output.GetTensorTypeAndShapeInfo();
  size_t text_emb_size = text_emb_info.GetElementCount();
  if (text_emb_size == 0) {
    SHERPA_ONNX_LOGE("Text encoder output is empty");
    SHERPA_ONNX_EXIT(-1);
  }
  auto *text_emb_data = text_enc_output.GetTensorMutableData<float>();
  std::vector<float> text_emb_vec(text_emb_data, text_emb_data + text_emb_size);
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
  int chunk_size = cfg.ae.base_chunk_size * cfg.ttl.chunk_compress_factor;
  int latent_len =
      static_cast<int>((wav_len_max + chunk_size - 1) / chunk_size);
  if (latent_len > kMaxLatentLen) {
    SHERPA_ONNX_LOGE(
        "Latent length (%d) exceeds maximum (%d), capping to prevent OOM",
        latent_len, kMaxLatentLen);
    latent_len = kMaxLatentLen;
  }

  int latent_dim = cfg.ttl.latent_dim * cfg.ttl.chunk_compress_factor;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 1.0f);
  size_t latent_total_size = static_cast<size_t>(bsz) *
                             static_cast<size_t>(latent_dim) *
                             static_cast<size_t>(latent_len);
  if (latent_total_size / static_cast<size_t>(bsz) /
          static_cast<size_t>(latent_dim) !=
      static_cast<size_t>(latent_len)) {
    SHERPA_ONNX_LOGE(
        "Latent total size overflow: bsz=%d, latent_dim=%d, latent_len=%d", bsz,
        latent_dim, latent_len);
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<float> xt_flat(latent_total_size);
  for (size_t i = 0; i < latent_total_size; ++i) {
    xt_flat[i] = dist(gen);
  }
  std::vector<float> latent_mask_flat;
  std::vector<int64_t> latent_mask_shape;
  GetLatentMaskFlat(wav_lengths, bsz, cfg.ae.base_chunk_size,
                    cfg.ttl.chunk_compress_factor, latent_len,
                    &latent_mask_flat, &latent_mask_shape);
  int64_t latent_mask_len = latent_mask_shape[2];
  if (latent_mask_len != latent_len) {
    SHERPA_ONNX_LOGE("Latent mask length mismatch: expected %d, got %ld",
                     latent_len, latent_mask_len);
    SHERPA_ONNX_EXIT(-1);
  }
  for (int b = 0; b < bsz; ++b) {
    const float *mask_batch = latent_mask_flat.data() + b * latent_mask_len;
    float *xt_batch = xt_flat.data() + b * latent_dim * latent_len;
    for (int d = 0; d < latent_dim; ++d) {
      float *xt_dim = xt_batch + d * latent_len;
      for (int t = 0; t < latent_len; ++t) {
        xt_dim[t] *= mask_batch[t];
      }
    }
  }

  std::vector<int64_t> latent_shape = {bsz, latent_dim, latent_len};
  std::vector<float> total_step_vec(bsz, static_cast<float>(num_steps));
  std::vector<int64_t> step_shape = {bsz};

  for (int32_t step = 0; step < num_steps; step++) {
    std::vector<float> current_step_vec(bsz, static_cast<float>(step));
    Ort::Value noisy_latent_tensor = Ort::Value::CreateTensor<float>(
        memory_info, xt_flat.data(), xt_flat.size(), latent_shape.data(),
        latent_shape.size());
    Ort::Value text_emb_tensor = Ort::Value::CreateTensor<float>(
        memory_info, text_emb_vec.data(), text_emb_vec.size(),
        text_emb_shape.data(), text_emb_shape.size());
    Ort::Value style_ttl_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(style.ttl_data.data()),
        style.ttl_data.size(), style.ttl_shape.data(), style.ttl_shape.size());
    Ort::Value text_mask_tensor = Ort::Value::CreateTensor<float>(
        memory_info, text_mask_flat.data(), text_mask_flat.size(),
        text_mask_shape.data(), text_mask_shape.size());
    Ort::Value latent_mask_tensor = Ort::Value::CreateTensor<float>(
        memory_info, latent_mask_flat.data(), latent_mask_flat.size(),
        latent_mask_shape.data(), latent_mask_shape.size());
    Ort::Value total_step_tensor = Ort::Value::CreateTensor<float>(
        memory_info, total_step_vec.data(), total_step_vec.size(),
        step_shape.data(), step_shape.size());
    Ort::Value current_step_tensor = Ort::Value::CreateTensor<float>(
        memory_info, current_step_vec.data(), current_step_vec.size(),
        step_shape.data(), step_shape.size());

    std::vector<Ort::Value> vector_est_inputs;
    vector_est_inputs.push_back(std::move(noisy_latent_tensor));
    vector_est_inputs.push_back(std::move(text_emb_tensor));
    vector_est_inputs.push_back(std::move(style_ttl_tensor));
    vector_est_inputs.push_back(std::move(latent_mask_tensor));
    vector_est_inputs.push_back(std::move(text_mask_tensor));
    vector_est_inputs.push_back(std::move(current_step_tensor));
    vector_est_inputs.push_back(std::move(total_step_tensor));

    Ort::Value vector_est_output =
        model_->RunVectorEstimator(std::move(vector_est_inputs));
    auto vector_est_output_info = vector_est_output.GetTensorTypeAndShapeInfo();
    size_t denoised_size = vector_est_output_info.GetElementCount();
    if (denoised_size != latent_total_size) {
      SHERPA_ONNX_LOGE(
          "Denoised latent size mismatch at step %d: expected %zu, got %zu",
          step, latent_total_size, denoised_size);
      SHERPA_ONNX_EXIT(-1);
    }
    auto *denoised_data = vector_est_output.GetTensorMutableData<float>();
    std::memcpy(xt_flat.data(), denoised_data,
                latent_total_size * sizeof(float));
  }

  Ort::Value latent_tensor = Ort::Value::CreateTensor<float>(
      memory_info, xt_flat.data(), xt_flat.size(), latent_shape.data(),
      latent_shape.size());
  std::vector<Ort::Value> vocoder_inputs;
  vocoder_inputs.push_back(std::move(latent_tensor));
  Ort::Value vocoder_output = model_->RunVocoder(std::move(vocoder_inputs));
  auto wav_info = vocoder_output.GetTensorTypeAndShapeInfo();
  auto wav_shape = wav_info.GetShape();
  size_t wav_size = wav_info.GetElementCount();
  if (wav_size == 0) {
    SHERPA_ONNX_LOGE("Vocoder output is empty");
    SHERPA_ONNX_EXIT(-1);
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
    result.samples.reserve(static_cast<size_t>(
        std::accumulate(wav_lengths.begin(), wav_lengths.end(), int64_t(0))));
    for (int b = 0; b < bsz; ++b) {
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
    float max_abs_val = *std::max_element(
        result.samples.begin(), result.samples.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });
    float min_abs_val = *std::min_element(
        result.samples.begin(), result.samples.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });
    SHERPA_ONNX_LOGE("Audio samples: %zu, min_abs=%.6f, max_abs=%.6f",
                     result.samples.size(), min_abs_val, max_abs_val);
  }
  result.sample_rate = cfg.ae.sample_rate;
  return result;
}

GeneratedAudio OfflineTtsSupertonicImpl::ProcessChunksAndConcatenate(
    const std::vector<std::string> &text_chunks, const std::string &lang,
    const SupertonicStyle &style, int32_t num_steps, float speed,
    float silence_duration, GeneratedAudioCallback callback) const {
  GeneratedAudio result;
  std::vector<float> wav_cat;
  int32_t num_chunks = static_cast<int32_t>(text_chunks.size());
  for (int32_t i = 0; i < num_chunks; ++i) {
    auto chunk_result =
        Process({text_chunks[i]}, {lang}, style, num_steps, speed);
    if (wav_cat.empty()) {
      wav_cat = chunk_result.samples;
    } else {
      int silence_len =
          static_cast<int>(silence_duration * chunk_result.sample_rate);
      std::vector<float> silence(silence_len, 0.0f);
      wav_cat.insert(wav_cat.end(), silence.begin(), silence.end());
      wav_cat.insert(wav_cat.end(), chunk_result.samples.begin(),
                     chunk_result.samples.end());
    }
    if (callback) {
      float progress =
          static_cast<float>(i + 1) / static_cast<float>(num_chunks);
      callback(chunk_result.samples.data(), chunk_result.samples.size(),
               progress);
    }
  }
  result.samples = std::move(wav_cat);
  result.sample_rate = model_->GetSampleRate();
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
        style.ttl_shape[0], style.dp_shape[0]);
    SHERPA_ONNX_EXIT(-1);
  }
  full_style_ = std::move(style);
}

SupertonicStyle OfflineTtsSupertonicImpl::GetStyleForSid(int64_t sid) const {
  if (num_speakers_ == 0) {
    SHERPA_ONNX_LOGE("No voice style loaded");
    SHERPA_ONNX_EXIT(-1);
  }
  if (num_speakers_ == 1) {
    return full_style_;
  }
  int32_t s = static_cast<int32_t>(sid);
  if (s < 0) {
    s = 0;
  }
  if (s >= num_speakers_) {
    s = num_speakers_ - 1;
  }
  const SupertonicStyle &full = full_style_;
  int64_t ttl_d1 = full.ttl_shape[1];
  int64_t ttl_d2 = full.ttl_shape[2];
  int64_t dp_d1 = full.dp_shape[1];
  int64_t dp_d2 = full.dp_shape[2];
  size_t ttl_slice = static_cast<size_t>(ttl_d1 * ttl_d2);
  size_t dp_slice = static_cast<size_t>(dp_d1 * dp_d2);
  size_t ttl_offset = static_cast<size_t>(s) * ttl_slice;
  size_t dp_offset = static_cast<size_t>(s) * dp_slice;
  SupertonicStyle out;
  out.ttl_shape = {1, ttl_d1, ttl_d2};
  out.dp_shape = {1, dp_d1, dp_d2};
  out.ttl_data.assign(full.ttl_data.begin() + ttl_offset,
                      full.ttl_data.begin() + ttl_offset + ttl_slice);
  out.dp_data.assign(full.dp_data.begin() + dp_offset,
                     full.dp_data.begin() + dp_offset + dp_slice);
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
