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
#include <cstring>
#include <fstream>
#include <functional>
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

#include "nlohmann/json.hpp"
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

using json = nlohmann::json;

namespace {

// Minimum duration (in seconds) to prevent zero-length audio
constexpr float kMinDuration = 0.1f;

// Maximum latent length to prevent excessive memory allocation and OOM.
constexpr int kMaxLatentLen = 10000;

template <typename T>
static void Flatten3DArray(
    const std::vector<std::vector<std::vector<T>>> &array,
    std::vector<T> *flat) {
  flat->clear();
  size_t total_size = 0;
  for (const auto &batch : array) {
    for (const auto &row : batch) {
      total_size += row.size();
    }
  }
  flat->resize(total_size);
  T *dest = flat->data();
  for (const auto &batch : array) {
    for (const auto &row : batch) {
      dest = std::copy(row.begin(), row.end(), dest);
    }
  }
}

template <typename T>
static void Flatten2DArray(const std::vector<std::vector<T>> &array,
                           std::vector<T> *flat) {
  flat->clear();
  size_t total_size = 0;
  for (const auto &row : array) {
    total_size += row.size();
  }
  flat->resize(total_size);
  T *dest = flat->data();
  for (const auto &row : array) {
    dest = std::copy(row.begin(), row.end(), dest);
  }
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
  sherpa_onnx::LengthToMaskFlat(latent_lengths, bsz, latent_len, mask_flat,
                                mask_shape);
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
    std::string para = sherpa_onnx::Trim(*iter);
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
    std::sregex_iterator match_end;
    size_t last_pos = 0;
    for (; match_iter != match_end; ++match_iter) {
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
    std::string current_chunk = "";
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
          chunks.push_back(sherpa_onnx::Trim(current_chunk));
        }
        current_chunk = sentence;
      }
    }
    if (!current_chunk.empty()) {
      chunks.push_back(sherpa_onnx::Trim(current_chunk));
    }
  }
  if (chunks.empty()) {
    chunks.push_back(sherpa_onnx::Trim(text));
  }
  return chunks;
}

static SupertonicStyle ParseVoiceStyleFromJson(const json &j) {
  if (j.find("style_ttl") == j.end() || j.find("style_dp") == j.end()) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style file: missing 'style_ttl' or 'style_dp'");
    SHERPA_ONNX_EXIT(-1);
  }
  auto ttl_dims = j["style_ttl"]["dims"].get<std::vector<int64_t>>();
  auto dp_dims = j["style_dp"]["dims"].get<std::vector<int64_t>>();
  // Validate dims: must not be empty and all values must be positive
  if (ttl_dims.empty()) {
    SHERPA_ONNX_LOGE("Invalid voice style: ttl_dims is empty");
    SHERPA_ONNX_EXIT(-1);
  }
  if (dp_dims.empty()) {
    SHERPA_ONNX_LOGE("Invalid voice style: dp_dims is empty");
    SHERPA_ONNX_EXIT(-1);
  }
  for (int64_t d : ttl_dims) {
    if (d <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid voice style: ttl_dims contains non-positive value (%ld)", d);
      SHERPA_ONNX_EXIT(-1);
    }
  }
  for (int64_t d : dp_dims) {
    if (d <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid voice style: dp_dims contains non-positive value (%ld)", d);
      SHERPA_ONNX_EXIT(-1);
    }
  }
  auto ttl_data_nested =
      j["style_ttl"]["data"]
          .get<std::vector<std::vector<std::vector<float>>>>();
  std::vector<float> ttl_data;
  Flatten3DArray(ttl_data_nested, &ttl_data);
  auto dp_data_nested =
      j["style_dp"]["data"].get<std::vector<std::vector<std::vector<float>>>>();
  std::vector<float> dp_data;
  Flatten3DArray(dp_data_nested, &dp_data);
  // Validate dims match data size with overflow protection
  int64_t ttl_expected_size = 1;
  for (int64_t d : ttl_dims) {
    if (ttl_expected_size > INT64_MAX / d) {
      SHERPA_ONNX_LOGE(
          "Invalid voice style: ttl_dims product overflow (dim=%ld, "
          "current=%ld)",
          d, ttl_expected_size);
      SHERPA_ONNX_EXIT(-1);
    }
    ttl_expected_size *= d;
  }
  if (static_cast<int64_t>(ttl_data.size()) != ttl_expected_size) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style: ttl_dims product (%ld) != ttl_data size (%zu)",
        ttl_expected_size, ttl_data.size());
    SHERPA_ONNX_EXIT(-1);
  }
  int64_t dp_expected_size = 1;
  for (int64_t d : dp_dims) {
    if (dp_expected_size > INT64_MAX / d) {
      SHERPA_ONNX_LOGE(
          "Invalid voice style: dp_dims product overflow (dim=%ld, "
          "current=%ld)",
          d, dp_expected_size);
      SHERPA_ONNX_EXIT(-1);
    }
    dp_expected_size *= d;
  }
  if (static_cast<int64_t>(dp_data.size()) != dp_expected_size) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style: dp_dims product (%ld) != dp_data size (%zu)",
        dp_expected_size, dp_data.size());
    SHERPA_ONNX_EXIT(-1);
  }
  SupertonicStyle style;
  style.ttl_data = std::move(ttl_data);
  style.dp_data = std::move(dp_data);
  style.ttl_shape = std::move(ttl_dims);
  style.dp_shape = std::move(dp_dims);
  return style;
}

static SupertonicStyle LoadVoiceStylesImpl(
    const std::vector<std::string> &voice_style_paths,
    std::function<nlohmann::json(const std::string &)> load_json_fn) {
  if (voice_style_paths.empty()) {
    SHERPA_ONNX_LOGE("Empty voice style paths");
    SHERPA_ONNX_EXIT(-1);
  }
  int32_t bsz = static_cast<int32_t>(voice_style_paths.size());
  json first_json = load_json_fn(voice_style_paths[0]);
  auto ttl_dims = first_json["style_ttl"]["dims"].get<std::vector<int64_t>>();
  auto dp_dims = first_json["style_dp"]["dims"].get<std::vector<int64_t>>();
  if (ttl_dims.size() != 3 || dp_dims.size() != 3) {
    SHERPA_ONNX_LOGE(
        "Invalid voice style dimensions: ttl_dims=%zu, dp_dims=%zu",
        ttl_dims.size(), dp_dims.size());
    SHERPA_ONNX_EXIT(-1);
  }
  int64_t ttl_dim1 = ttl_dims[1];
  int64_t ttl_dim2 = ttl_dims[2];
  int64_t dp_dim1 = dp_dims[1];
  int64_t dp_dim2 = dp_dims[2];
  size_t ttl_size = static_cast<size_t>(bsz) * static_cast<size_t>(ttl_dim1) *
                    static_cast<size_t>(ttl_dim2);
  size_t dp_size = static_cast<size_t>(bsz) * static_cast<size_t>(dp_dim1) *
                   static_cast<size_t>(dp_dim2);
  std::vector<float> ttl_flat(ttl_size);
  std::vector<float> dp_flat(dp_size);
  for (int32_t i = 0; i < bsz; ++i) {
    json j = load_json_fn(voice_style_paths[i]);
    // Validate and parse each style file using ParseVoiceStyleFromJson
    SupertonicStyle style = ParseVoiceStyleFromJson(j);
    // Verify dims[1] and dims[2] match the first file
    if (style.ttl_shape.size() != 3 || style.dp_shape.size() != 3) {
      SHERPA_ONNX_LOGE(
          "Invalid voice style[%d]: ttl_shape.size()=%zu, dp_shape.size()=%zu",
          i, style.ttl_shape.size(), style.dp_shape.size());
      SHERPA_ONNX_EXIT(-1);
    }
    if (style.ttl_shape[1] != ttl_dim1 || style.ttl_shape[2] != ttl_dim2) {
      SHERPA_ONNX_LOGE(
          "Invalid voice style[%d]: ttl_shape[1,2]={%ld,%ld} != expected "
          "{%ld,%ld}",
          i, style.ttl_shape[1], style.ttl_shape[2], ttl_dim1, ttl_dim2);
      SHERPA_ONNX_EXIT(-1);
    }
    if (style.dp_shape[1] != dp_dim1 || style.dp_shape[2] != dp_dim2) {
      SHERPA_ONNX_LOGE(
          "Invalid voice style[%d]: dp_shape[1,2]={%ld,%ld} != expected "
          "{%ld,%ld}",
          i, style.dp_shape[1], style.dp_shape[2], dp_dim1, dp_dim2);
      SHERPA_ONNX_EXIT(-1);
    }
    // Verify data size matches expected dimensions
    size_t expected_ttl_size =
        static_cast<size_t>(ttl_dim1) * static_cast<size_t>(ttl_dim2);
    if (style.ttl_data.size() != expected_ttl_size) {
      SHERPA_ONNX_LOGE(
          "Invalid voice style[%d]: ttl_data size (%zu) != expected (%zu)", i,
          style.ttl_data.size(), expected_ttl_size);
      SHERPA_ONNX_EXIT(-1);
    }
    size_t expected_dp_size =
        static_cast<size_t>(dp_dim1) * static_cast<size_t>(dp_dim2);
    if (style.dp_data.size() != expected_dp_size) {
      SHERPA_ONNX_LOGE(
          "Invalid voice style[%d]: dp_data size (%zu) != expected (%zu)", i,
          style.dp_data.size(), expected_dp_size);
      SHERPA_ONNX_EXIT(-1);
    }
    size_t ttl_offset = static_cast<size_t>(i) * static_cast<size_t>(ttl_dim1) *
                        static_cast<size_t>(ttl_dim2);
    std::copy(style.ttl_data.begin(), style.ttl_data.end(),
              ttl_flat.begin() + ttl_offset);
    size_t dp_offset = static_cast<size_t>(i) * static_cast<size_t>(dp_dim1) *
                       static_cast<size_t>(dp_dim2);
    std::copy(style.dp_data.begin(), style.dp_data.end(),
              dp_flat.begin() + dp_offset);
  }
  SupertonicStyle style;
  style.ttl_data = std::move(ttl_flat);
  style.dp_data = std::move(dp_flat);
  style.ttl_shape = {bsz, ttl_dim1, ttl_dim2};
  style.dp_shape = {bsz, dp_dim1, dp_dim2};
  return style;
}

}  // namespace

OfflineTtsSupertonicImpl::OfflineTtsSupertonicImpl(
    const OfflineTtsConfig &config)
    : config_(config),
      model_(std::make_unique<OfflineTtsSupertonicModel>(config.model)),
      text_processor_(std::make_unique<SupertonicUnicodeProcessor>(
          ResolveAbsolutePath(config.model.supertonic.model_dir) +
          "/unicode_indexer.json")) {}

template <typename Manager>
OfflineTtsSupertonicImpl::OfflineTtsSupertonicImpl(
    Manager *mgr, const OfflineTtsConfig &config)
    : config_(config),
      model_(std::make_unique<OfflineTtsSupertonicModel>(mgr, config.model)),
      text_processor_(std::make_unique<SupertonicUnicodeProcessor>(
          mgr, config.model.supertonic.model_dir + "/unicode_indexer.json")) {}

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
  if (config.sid != 0 && config_.model.debug) {
    SHERPA_ONNX_LOGE(
        "Supertonic model doesn't support speaker ID, ignoring sid=%d",
        config.sid);
  }
  float speed =
      config.speed > 0 ? config.speed : config_.model.supertonic.speed;
  int32_t num_steps = config.num_steps > 0 ? config.num_steps
                                           : config_.model.supertonic.num_steps;
  std::vector<std::string> text_list = SplitStringAndTrim(text, '|');
  if (text_list.empty()) {
    SHERPA_ONNX_LOGE("Input text is empty");
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<std::string> lang_list;
  std::string lang_str = config.GetExtraString("lang", "");
  std::string batch_str = config.GetExtraString("batch", "");
  bool batch_mode_flag = (batch_str == "1" || batch_str == "true");
  if (!lang_str.empty()) {
    lang_list = SplitStringAndTrim(lang_str, ',');
  } else {
    lang_list.resize(text_list.size(), "en");
  }
  if (lang_list.size() != text_list.size()) {
    if (lang_list.size() == 1) {
      lang_list.resize(text_list.size(), lang_list[0]);
    } else {
      SHERPA_ONNX_LOGE(
          "Number of languages (%zu) must match number of texts (%zu)",
          lang_list.size(), text_list.size());
      SHERPA_ONNX_EXIT(-1);
    }
  }
  std::vector<std::string> voice_style_paths =
      SplitStringAndTrim(config_.model.supertonic.voice_style, ',');
  if (voice_style_paths.size() != text_list.size()) {
    if (voice_style_paths.size() == 1) {
      voice_style_paths.resize(text_list.size(), voice_style_paths[0]);
    } else {
      SHERPA_ONNX_LOGE(
          "Number of voice styles (%zu) must match number of texts (%zu)",
          voice_style_paths.size(), text_list.size());
      SHERPA_ONNX_EXIT(-1);
    }
  }
  bool batch_mode = (text_list.size() > 1) || batch_mode_flag;
  if (batch_mode) {
    SupertonicStyle style = LoadVoiceStyles(voice_style_paths);
    return Process(text_list, lang_list, style, num_steps, speed);
  } else {
    std::string lang = lang_list[0];
    float silence_duration = 0.3f;
    size_t max_len =
        (lang == "ko")
            ? static_cast<size_t>(config_.model.supertonic.max_len_korean)
            : static_cast<size_t>(config_.model.supertonic.max_len_other);
    auto text_chunks = ChunkText(text_list[0], max_len);
    SupertonicStyle style = LoadVoiceStyle(voice_style_paths[0]);
    return ProcessChunksAndConcatenate(text_chunks, lang, style, num_steps,
                                       speed, silence_duration, callback);
  }
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
        "Number of texts (%d) must match number of style vectors (%d)", bsz,
        static_cast<int32_t>(style.ttl_shape[0]));
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<std::vector<int64_t>> text_ids;
  std::vector<std::vector<std::vector<float>>> text_mask;
  text_processor_->Process(text_list, lang_list, &text_ids, &text_mask);
  if (text_ids.empty() || text_mask.empty()) {
    SHERPA_ONNX_LOGE("Text processing failed: empty text_ids or text_mask");
    SHERPA_ONNX_EXIT(-1);
  }
  int64_t text_seq_len = static_cast<int64_t>(text_ids[0].size());
  int64_t text_mask_len = static_cast<int64_t>(text_mask[0][0].size());
  if (text_seq_len != text_mask_len) {
    SHERPA_ONNX_LOGE("Text sequence length mismatch: text_ids=%d, text_mask=%d",
                     static_cast<int32_t>(text_seq_len),
                     static_cast<int32_t>(text_mask_len));
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<int64_t> text_ids_shape = {bsz, text_seq_len};
  std::vector<int64_t> text_mask_shape = {bsz, 1, text_mask_len};
  std::vector<int64_t> text_ids_flat;
  text_ids_flat.reserve(bsz * text_seq_len);
  Flatten2DArray(text_ids, &text_ids_flat);
  std::vector<float> text_mask_flat;
  text_mask_flat.reserve(bsz * text_mask_len);
  Flatten3DArray(text_mask, &text_mask_flat);

  Ort::Value style_dp_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(style.dp_data.data()),
      style.dp_data.size(), style.dp_shape.data(), style.dp_shape.size());
  const char *dp_input_names[] = {"text_ids", "style_dp", "text_mask"};
  const char *dp_output_names[] = {"duration"};
  std::vector<Ort::Value> dp_inputs;
  dp_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
      memory_info, text_ids_flat.data(), text_ids_flat.size(),
      text_ids_shape.data(), text_ids_shape.size()));
  dp_inputs.push_back(std::move(style_dp_tensor));
  dp_inputs.push_back(Ort::Value::CreateTensor<float>(
      memory_info, text_mask_flat.data(), text_mask_flat.size(),
      text_mask_shape.data(), text_mask_shape.size()));
  auto dp_outputs = model_->GetDurationPredictorSession()->Run(
      Ort::RunOptions{nullptr}, dp_input_names, dp_inputs.data(),
      dp_inputs.size(), dp_output_names, 1);
  if (dp_outputs.empty()) {
    SHERPA_ONNX_LOGE("Duration predictor returned empty output");
    SHERPA_ONNX_EXIT(-1);
  }
  auto dp_output_info = dp_outputs[0].GetTensorTypeAndShapeInfo();
  size_t dp_element_count = dp_output_info.GetElementCount();
  if (dp_element_count != static_cast<size_t>(bsz)) {
    SHERPA_ONNX_LOGE(
        "Duration predictor output size mismatch: expected %d, got %zu", bsz,
        dp_element_count);
    SHERPA_ONNX_EXIT(-1);
  }
  auto *dur_data = dp_outputs[0].GetTensorMutableData<float>();
  std::vector<float> duration(dur_data, dur_data + bsz);
  for (auto &dur : duration) {
    dur /= speed;
    // Ensure minimum duration to avoid zero-length audio
    if (dur < kMinDuration) {
      dur = kMinDuration;
    }
  }

  const char *text_enc_input_names[] = {"text_ids", "style_ttl", "text_mask"};
  const char *text_enc_output_names[] = {"text_emb"};
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
  auto text_enc_outputs = model_->GetTextEncoderSession()->Run(
      Ort::RunOptions{nullptr}, text_enc_input_names, text_enc_inputs.data(),
      text_enc_inputs.size(), text_enc_output_names, 1);
  if (text_enc_outputs.empty()) {
    SHERPA_ONNX_LOGE("Text encoder returned empty output");
    SHERPA_ONNX_EXIT(-1);
  }
  auto text_emb_info = text_enc_outputs[0].GetTensorTypeAndShapeInfo();
  size_t text_emb_size = text_emb_info.GetElementCount();
  if (text_emb_size == 0) {
    SHERPA_ONNX_LOGE("Text encoder output is empty");
    SHERPA_ONNX_EXIT(-1);
  }
  auto *text_emb_data = text_enc_outputs[0].GetTensorMutableData<float>();
  std::vector<float> text_emb_vec(text_emb_data, text_emb_data + text_emb_size);
  auto text_emb_shape = text_emb_info.GetShape();

  float wav_len_max =
      *std::max_element(duration.begin(), duration.end()) * cfg.ae.sample_rate;
  std::vector<int64_t> wav_lengths;
  wav_lengths.reserve(bsz);
  for (float d : duration) {
    int64_t wav_len = static_cast<int64_t>(d * cfg.ae.sample_rate);
    // Ensure minimum wav_length >= 1
    if (wav_len < 1) {
      wav_len = 1;
    }
    wav_lengths.push_back(wav_len);
  }
  int chunk_size = cfg.ae.base_chunk_size * cfg.ttl.chunk_compress_factor;
  int latent_len =
      static_cast<int>((wav_len_max + chunk_size - 1) / chunk_size);
  // Cap latent_len to prevent excessive memory allocation
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
  // Check for overflow
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
  const char *vector_est_input_names[] = {
      "noisy_latent", "text_emb",   "style_ttl",   "text_mask",
      "latent_mask",  "total_step", "current_step"};
  const char *vector_est_output_names[] = {"denoised_latent"};

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
    vector_est_inputs.push_back(std::move(text_mask_tensor));
    vector_est_inputs.push_back(std::move(latent_mask_tensor));
    vector_est_inputs.push_back(std::move(total_step_tensor));
    vector_est_inputs.push_back(std::move(current_step_tensor));

    auto vector_est_outputs = model_->GetVectorEstimatorSession()->Run(
        Ort::RunOptions{nullptr}, vector_est_input_names,
        vector_est_inputs.data(), vector_est_inputs.size(),
        vector_est_output_names, 1);

    if (vector_est_outputs.empty()) {
      SHERPA_ONNX_LOGE("Vector estimator returned empty output at step %d",
                       step);
      SHERPA_ONNX_EXIT(-1);
    }
    auto vector_est_output_info =
        vector_est_outputs[0].GetTensorTypeAndShapeInfo();
    size_t denoised_size = vector_est_output_info.GetElementCount();
    if (denoised_size != latent_total_size) {
      SHERPA_ONNX_LOGE(
          "Denoised latent size mismatch at step %d: expected %zu, got %zu",
          step, latent_total_size, denoised_size);
      SHERPA_ONNX_EXIT(-1);
    }
    auto *denoised_data = vector_est_outputs[0].GetTensorMutableData<float>();
    std::memcpy(xt_flat.data(), denoised_data,
                latent_total_size * sizeof(float));
  }

  Ort::Value latent_tensor = Ort::Value::CreateTensor<float>(
      memory_info, xt_flat.data(), xt_flat.size(), latent_shape.data(),
      latent_shape.size());
  const char *vocoder_input_names[] = {"latent"};
  const char *vocoder_output_names[] = {"wav_tts"};
  std::vector<Ort::Value> vocoder_inputs;
  vocoder_inputs.push_back(std::move(latent_tensor));
  auto vocoder_outputs = model_->GetVocoderSession()->Run(
      Ort::RunOptions{nullptr}, vocoder_input_names, vocoder_inputs.data(),
      vocoder_inputs.size(), vocoder_output_names, 1);
  if (vocoder_outputs.empty()) {
    SHERPA_ONNX_LOGE("Vocoder returned empty output");
    SHERPA_ONNX_EXIT(-1);
  }

  auto wav_info = vocoder_outputs[0].GetTensorTypeAndShapeInfo();
  auto wav_shape = wav_info.GetShape();
  size_t wav_size = wav_info.GetElementCount();
  if (wav_size == 0) {
    SHERPA_ONNX_LOGE("Vocoder output is empty");
    SHERPA_ONNX_EXIT(-1);
  }

  auto *wav_data = vocoder_outputs[0].GetTensorMutableData<float>();
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
  if (wav_shape.size() == 2 && wav_shape[0] == bsz) {
    int64_t samples_per_batch = wav_shape[1];
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
  } else if (wav_shape.size() == 3 && wav_shape[0] == bsz &&
             wav_shape[1] == 1) {
    int64_t samples_per_batch = wav_shape[2];
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
  } else if (wav_shape.size() == 1) {
    result.samples.assign(wav_data, wav_data + wav_size);
  } else if (wav_shape.size() == 2 && wav_shape[0] == 1) {
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

SupertonicStyle OfflineTtsSupertonicImpl::LoadVoiceStyle(
    const std::string &voice_style_path) const {
  return ParseVoiceStyleFromJson(
      sherpa_onnx::LoadJsonFromFile(voice_style_path));
}

SupertonicStyle OfflineTtsSupertonicImpl::LoadVoiceStyles(
    const std::vector<std::string> &voice_style_paths) const {
  return LoadVoiceStylesImpl(voice_style_paths, [](const std::string &path) {
    return sherpa_onnx::LoadJsonFromFile(path);
  });
}

template <typename Manager>
SupertonicStyle OfflineTtsSupertonicImpl::LoadVoiceStyle(
    Manager *mgr, const std::string &voice_style_path) const {
  return ParseVoiceStyleFromJson(
      sherpa_onnx::LoadJsonFromFile(mgr, voice_style_path));
}

template <typename Manager>
SupertonicStyle OfflineTtsSupertonicImpl::LoadVoiceStyles(
    Manager *mgr, const std::vector<std::string> &voice_style_paths) const {
  return LoadVoiceStylesImpl(voice_style_paths, [mgr](const std::string &path) {
    return sherpa_onnx::LoadJsonFromFile(mgr, path);
  });
}

#if __ANDROID_API__ >= 9
template OfflineTtsSupertonicImpl::OfflineTtsSupertonicImpl(
    AAssetManager *mgr, const OfflineTtsConfig &config);
template SupertonicStyle OfflineTtsSupertonicImpl::LoadVoiceStyle(
    AAssetManager *mgr, const std::string &voice_style_path) const;
template SupertonicStyle OfflineTtsSupertonicImpl::LoadVoiceStyles(
    AAssetManager *mgr,
    const std::vector<std::string> &voice_style_paths) const;
#endif

#if __OHOS__
template OfflineTtsSupertonicImpl::OfflineTtsSupertonicImpl(
    NativeResourceManager *mgr, const OfflineTtsConfig &config);
template SupertonicStyle OfflineTtsSupertonicImpl::LoadVoiceStyle(
    NativeResourceManager *mgr, const std::string &voice_style_path) const;
template SupertonicStyle OfflineTtsSupertonicImpl::LoadVoiceStyles(
    NativeResourceManager *mgr,
    const std::vector<std::string> &voice_style_paths) const;
#endif

}  // namespace sherpa_onnx
