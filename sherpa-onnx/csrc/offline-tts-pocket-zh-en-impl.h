// sherpa-onnx/csrc/offline-tts-pocket-zh-en-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Please refer to
// https://modelscope.cn/models/dengcunqin/pocket-tts-zh-en
// for the model files.

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_IMPL_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/fst-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/normal-data-generator.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-pocket-zh-en-model.h"
#include "sherpa-onnx/csrc/pocket-zh-en-lexicon.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsPocketZhEnImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsPocketZhEnImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsPocketZhEnModel>(config.model)),
        lexicon_(std::make_unique<PocketZhEnLexicon>(
            config.model.pocket_zh_en.lexicon, config.model.debug)) {
    cache_.SetCapacity(
        config.model.pocket_zh_en.voice_embedding_cache_capacity);

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule fst: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
#endif
        }
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
      }
    }

    if (!config.rule_fars.empty()) {
      if (config.model.debug) {
        SHERPA_ONNX_LOGE("Loading FST archives");
      }
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);

      tn_list_.reserve(files.size() + tn_list_.size());

      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule far: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
#endif
        }
        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(f));
        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }
      }

      if (config.model.debug) {
        SHERPA_ONNX_LOGE("FST archives loaded!");
      }
    }
  }

  template <typename Manager>
  OfflineTtsPocketZhEnImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsPocketZhEnModel>(mgr, config.model)),
        lexicon_(std::make_unique<PocketZhEnLexicon>(
            mgr, config.model.pocket_zh_en.lexicon, config.model.debug)) {
    cache_.SetCapacity(
        config.model.pocket_zh_en.voice_embedding_cache_capacity);

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule fst: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
#endif
        }
        auto buf = ReadFile(mgr, f);
        std::istringstream is(std::string(buf.data(), buf.size()));
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(is));
      }
    }

    if (!config.rule_fars.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);
      tn_list_.reserve(files.size() + tn_list_.size());

      for (const auto &f : files) {
        if (config.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("rule far: %{public}s", f.c_str());
#else
          SHERPA_ONNX_LOGE("rule far: %s", f.c_str());
#endif
        }

        auto buf = ReadFile(mgr, f);

        auto fsts = ReadFstsFromFar(buf);
        for (auto &r : fsts) {
          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }
      }  // for (const auto &f : files)
    }  // if (!config.rule_fars.empty())
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  int32_t NumSpeakers() const override { return 1; }

  /**
   *
   * Supported extra parameters:
   *
   *  - max_frames, int, default 0 (auto-calculate from text length)
   *  - frames_after_eos, int, default 0
   *  - temperature, float, default 0.0
   *  - max_reference_audio_len, float, default 10, in seconds
   *  - seed, int, default 0
   */
  GeneratedAudio Generate(
      const std::string &_text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback = nullptr) const override {
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("%s", gen_config.ToString().c_str());
    }

    std::string text = _text;

    // Apply text normalization if rule FSTs/FARs are provided
    if (!tn_list_.empty()) {
      for (const auto &tn : tn_list_) {
        text = tn->Normalize(text);
        if (config_.model.debug) {
#if __OHOS__
          SHERPA_ONNX_LOGE("After normalizing: %{public}s", text.c_str());
#else
          SHERPA_ONNX_LOGE("After normalizing: %s", text.c_str());
#endif
        }
      }
    }

    // Get voice embedding
    Ort::Value voice_embedding = GetVoiceEmbedding(gen_config);
    if (!voice_embedding) {
      return {};
    }

    // Convert text to token IDs
    std::vector<int32_t> token_ids = lexicon_->ConvertTextToTokenIds(text);
    if (token_ids.empty()) {
      SHERPA_ONNX_LOGE("Empty token IDs for text: %s", text.c_str());
      return {};
    }

    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("Token IDs count: %d",
                       static_cast<int32_t>(token_ids.size()));
    }

    GeneratedAudio result;
    result.sample_rate = SampleRate();

    bool should_continue = true;

    GeneratedAudio cur = GenerateSingleSentence(
        token_ids, gen_config, View(&voice_embedding), should_continue,
        callback);

    if (cur.samples.empty()) {
      return {};
    }

    result.samples = std::move(cur.samples);

    float silence_scale = gen_config.silence_scale;
    if (silence_scale != 1) {
      result = result.ScaleSilence(silence_scale);
    }

    return result;
  }

 private:
  static size_t ComputeHash(const float *p, size_t n) {
    size_t hash = 0;

    auto hash_combine = [](size_t &seed, size_t value) {
      seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    };

    hash_combine(hash, n);

    for (size_t i = 0; i < n; ++i) {
      uint32_t bits;
      std::memcpy(&bits, &p[i], sizeof(float));
      hash_combine(hash, bits);
    }

    return hash;
  }

  GeneratedAudio GenerateSingleSentence(
      const std::vector<int32_t> &token_ids, const GenerationConfig &gen_config,
      Ort::Value voice_embedding, bool &should_continue,
      GeneratedAudioCallback callback) const {
    const auto &meta = model_->GetMetaData();

    // Get runtime parameters from gen_config.extra (matching Python demo defaults)
    int32_t max_frames = gen_config.GetExtraInt("max_frames", 0);
    int32_t frames_after_eos = gen_config.GetExtraInt("frames_after_eos", 0);
    float temperature = gen_config.GetExtraFloat("temperature", 0.0f);
    int32_t seed = gen_config.GetExtraInt("seed", 0);

    // Calculate max_frames from text length if not specified
    // tokens_per_second_estimate = 3.0, gen_seconds_padding = 2.0, frame_rate = 12.5
    if (max_frames <= 0) {
      float tokens_per_sec = 3.0f;
      float gen_padding = 2.0f;
      float frame_rate = 12.5f;
      float estimated_seconds =
          static_cast<float>(token_ids.size()) / tokens_per_sec + gen_padding;
      max_frames =
          static_cast<int32_t>(std::ceil(estimated_seconds * frame_rate));
    }

    // Create random number generator
    NormalDataGenerator normal_gen(0, std::sqrt(temperature), seed);

    // Create initial tensors using model helpers
    auto latent = model_->CreateZeroLatent();
    auto is_bos = model_->CreateBosFlag();
    auto cond_gates = model_->CreateCondGates();
    auto text_gates = model_->CreateTextGates();
    auto latent_gates = model_->CreateLatentGates();
    auto zero_noise = model_->CreateZeroNoise();
    auto flow_kv = model_->CreateEmptyFlowKv();
    auto flow_offset = model_->CreateZeroFlowOffset();
    auto mimi_kv = model_->CreateZeroMimiKv();
    auto mimi_offset = model_->CreateZeroMimiOffset();
    auto mimi_conv = model_->CreateZeroMimiConv();
    auto decode_steps = model_->CreateDecodeSteps();

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // ===== Voice prefill =====

    // Get voice embedding shape
    auto voice_shape = voice_embedding.GetTensorTypeAndShapeInfo().GetShape();
    int64_t voice_seq = voice_shape[1];  // number of frames

    // Create empty tokens for voice prefill [1, voice_seq]
    std::vector<int64_t> voice_tokens_data(voice_seq, 0);
    std::vector<int64_t> voice_tokens_shape = {1, voice_seq};
    Ort::Value voice_tokens = Ort::Value::CreateTensor(
        memory_info, voice_tokens_data.data(), voice_tokens_data.size(),
        voice_tokens_shape.data(), voice_tokens_shape.size());

    // Create latent for voice prefill [1, voice_seq, latent_dim]
    std::vector<float> voice_latent_data(voice_seq * meta.latent_dim, 0.0f);
    std::vector<int64_t> voice_latent_shape = {1, voice_seq, meta.latent_dim};
    Ort::Value voice_latent = Ort::Value::CreateTensor(
        memory_info, voice_latent_data.data(), voice_latent_data.size(),
        voice_latent_shape.data(), voice_latent_shape.size());

    // Create is_bos for voice prefill [1, voice_seq, 1]
    std::vector<float> voice_bos_data(voice_seq, 0.0f);
    std::vector<int64_t> voice_bos_shape = {1, voice_seq, 1};
    Ort::Value voice_bos = Ort::Value::CreateTensor(
        memory_info, voice_bos_data.data(), voice_bos_data.size(),
        voice_bos_shape.data(), voice_bos_shape.size());

    // Run voice prefill with COND gates
    std::vector<Ort::Value> voice_inputs;
    voice_inputs.reserve(12);
    voice_inputs.push_back(std::move(voice_tokens));
    voice_inputs.push_back(std::move(voice_latent));
    voice_inputs.push_back(std::move(voice_bos));
    voice_inputs.push_back(std::move(voice_embedding));
    voice_inputs.push_back(std::move(cond_gates));
    voice_inputs.push_back(View(&zero_noise));
    voice_inputs.push_back(View(&flow_kv));
    voice_inputs.push_back(View(&flow_offset));
    voice_inputs.push_back(View(&mimi_kv));
    voice_inputs.push_back(View(&mimi_offset));
    voice_inputs.push_back(View(&mimi_conv));
    voice_inputs.push_back(View(&decode_steps));

    auto voice_outputs = model_->RunStep(std::move(voice_inputs));

    // Update flow_kv from voice prefill
    flow_kv = std::move(voice_outputs[3]);

    // ===== Text prefill =====
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("Text prefill with %d tokens",
                       static_cast<int32_t>(token_ids.size()));
    }

    int64_t text_seq = static_cast<int64_t>(token_ids.size());

    // Create tokens tensor [1, text_seq]
    std::vector<int64_t> token_ids_i64(token_ids.begin(), token_ids.end());
    std::vector<int64_t> tokens_shape = {1, text_seq};
    Ort::Value tokens_tensor = Ort::Value::CreateTensor(
        memory_info, token_ids_i64.data(), token_ids_i64.size(),
        tokens_shape.data(), tokens_shape.size());

    // Create zeros latent for text prefill [1, text_seq, latent_dim]
    std::vector<float> text_latent_data(text_seq * meta.latent_dim, 0.0f);
    std::vector<int64_t> text_latent_shape = {1, text_seq, meta.latent_dim};
    Ort::Value text_latent = Ort::Value::CreateTensor(
        memory_info, text_latent_data.data(), text_latent_data.size(),
        text_latent_shape.data(), text_latent_shape.size());

    // Create zeros is_bos for text prefill [1, text_seq, 1]
    std::vector<float> text_bos_data(text_seq, 0.0f);
    std::vector<int64_t> text_bos_shape = {1, text_seq, 1};
    Ort::Value text_bos = Ort::Value::CreateTensor(
        memory_info, text_bos_data.data(), text_bos_data.size(),
        text_bos_shape.data(), text_bos_shape.size());

    // Create zeros cond for text prefill [1, text_seq, model_dim]
    std::vector<float> text_cond_data(text_seq * meta.model_dim, 0.0f);
    std::vector<int64_t> text_cond_shape = {1, text_seq, meta.model_dim};
    Ort::Value text_cond = Ort::Value::CreateTensor(
        memory_info, text_cond_data.data(), text_cond_data.size(),
        text_cond_shape.data(), text_cond_shape.size());

    // Accumulate flow_kv from voice prefill
    int32_t flow_kv_elem_size = meta.flow_layers * 2 * meta.flow_heads *
                                meta.flow_head_dim;
    std::vector<float> flow_kv_cache;
    {
      const float *data = flow_kv.GetTensorData<float>();
      auto shape = flow_kv.GetTensorTypeAndShapeInfo().GetShape();
      size_t size = 1;
      for (auto d : shape) size *= d;
      flow_kv_cache.assign(data, data + size);
    }

    // Create flow_offset for text prefill (current length of flow_kv cache)
    int64_t text_flow_offset = static_cast<int64_t>(
        flow_kv_cache.size() / flow_kv_elem_size);
    Ort::Value text_flow_offset_tensor = Ort::Value::CreateTensor(
        memory_info, &text_flow_offset, 1, nullptr, 0);

    // Run text prefill with TEXT gates
    std::vector<Ort::Value> text_inputs;
    text_inputs.reserve(12);
    text_inputs.push_back(std::move(tokens_tensor));
    text_inputs.push_back(std::move(text_latent));
    text_inputs.push_back(std::move(text_bos));
    text_inputs.push_back(std::move(text_cond));
    text_inputs.push_back(std::move(text_gates));
    text_inputs.push_back(View(&zero_noise));
    text_inputs.push_back(View(&flow_kv));
    text_inputs.push_back(std::move(text_flow_offset_tensor));
    text_inputs.push_back(View(&mimi_kv));
    text_inputs.push_back(View(&mimi_offset));
    text_inputs.push_back(View(&mimi_conv));
    text_inputs.push_back(View(&decode_steps));

    auto text_outputs = model_->RunStep(std::move(text_inputs));

    // Accumulate flow_kv from text prefill
    {
      const float *data = text_outputs[3].GetTensorData<float>();
      auto shape = text_outputs[3].GetTensorTypeAndShapeInfo().GetShape();
      size_t size = 1;
      for (auto d : shape) size *= d;
      flow_kv_cache.insert(flow_kv_cache.end(), data, data + size);
    }

    // Streaming generation
    std::vector<float> audio_samples;
    int32_t eos_frame = -1;
    int32_t total_frames = max_frames;  // default, updated if EOS detected
    std::vector<Ort::Value> outputs;

    // mimi_kv cache: sliding window of last mimi_kv_len entries
    int32_t mimi_kv_elem_size = meta.mimi_layers * 2 * meta.mimi_heads *
                                meta.mimi_head_dim;
    std::vector<float> mimi_kv_cache;  // grows, last mimi_kv_len used as input

    for (int32_t frame = 0; frame < max_frames && should_continue; ++frame) {
      // Create random noise
      std::vector<float> noise_data(meta.latent_dim);
      normal_gen.Fill(noise_data.data(), noise_data.size());
      std::vector<int64_t> noise_shape = {1, meta.latent_dim};
      Ort::Value noise_tensor = Ort::Value::CreateTensor(
          memory_info, noise_data.data(), noise_data.size(), noise_shape.data(),
          noise_shape.size());

      // Create tokens tensor [1, 1] with zero
      int64_t zero_token = 0;
      std::vector<int64_t> step_tokens_shape = {1, 1};
      Ort::Value step_tokens = Ort::Value::CreateTensor(
          memory_info, &zero_token, 1, step_tokens_shape.data(),
          step_tokens_shape.size());

      // Create zeros cond tensor [1, 1, model_dim]
      std::vector<float> step_cond_data(meta.model_dim, 0.0f);
      std::vector<int64_t> step_cond_shape = {1, 1, meta.model_dim};
      Ort::Value step_cond = Ort::Value::CreateTensor(
          memory_info, step_cond_data.data(), step_cond_data.size(),
          step_cond_shape.data(), step_cond_shape.size());

      // Create mimi_kv input: sliding window of last mimi_kv_len entries
      // If cache is shorter, pad with zeros at the beginning
      int32_t mimi_kv_rows = static_cast<int32_t>(
          mimi_kv_cache.size() / mimi_kv_elem_size);
      std::vector<float> mimi_kv_input(meta.mimi_kv_len * mimi_kv_elem_size,
                                       0.0f);
      if (mimi_kv_rows >= meta.mimi_kv_len) {
        // Copy last mimi_kv_len rows
        int32_t src_start = (mimi_kv_rows - meta.mimi_kv_len) * mimi_kv_elem_size;
        std::copy(mimi_kv_cache.begin() + src_start,
                  mimi_kv_cache.end(),
                  mimi_kv_input.begin());
      } else if (mimi_kv_rows > 0) {
        // Pad with zeros at beginning, copy all available
        int32_t dst_start = (meta.mimi_kv_len - mimi_kv_rows) * mimi_kv_elem_size;
        std::copy(mimi_kv_cache.begin(),
                  mimi_kv_cache.end(),
                  mimi_kv_input.begin() + dst_start);
      }
      std::vector<int64_t> mimi_kv_shape = {
          meta.mimi_kv_len, meta.mimi_layers, 2, 1,
          meta.mimi_heads, meta.mimi_head_dim};
      Ort::Value mimi_kv_tensor = Ort::Value::CreateTensor(
          memory_info, mimi_kv_input.data(), mimi_kv_input.size(),
          mimi_kv_shape.data(), mimi_kv_shape.size());

      // Create flow_kv input from cache (all accumulated entries)
      int64_t flow_kv_len = static_cast<int64_t>(
          flow_kv_cache.size() / flow_kv_elem_size);
      std::vector<int64_t> flow_kv_shape = {
          flow_kv_len, meta.flow_layers, 2, 1,
          meta.flow_heads, meta.flow_head_dim};
      Ort::Value flow_kv_tensor = Ort::Value::CreateTensor(
          memory_info,
          flow_kv_cache.empty() ? nullptr : flow_kv_cache.data(),
          flow_kv_cache.size(),
          flow_kv_shape.data(), flow_kv_shape.size());

      // Create flow_offset tensor (current length of flow_kv cache)
      int64_t flow_offset_val = flow_kv_len;
      Ort::Value flow_offset_tensor = Ort::Value::CreateTensor(
          memory_info, &flow_offset_val, 1, nullptr, 0);

      // Run step
      std::vector<Ort::Value> step_inputs;
      step_inputs.reserve(12);
      step_inputs.push_back(std::move(step_tokens));
      step_inputs.push_back(std::move(latent));
      step_inputs.push_back(std::move(is_bos));
      step_inputs.push_back(std::move(step_cond));
      step_inputs.push_back(std::move(latent_gates));
      step_inputs.push_back(std::move(noise_tensor));
      step_inputs.push_back(std::move(flow_kv_tensor));
      step_inputs.push_back(std::move(flow_offset_tensor));
      step_inputs.push_back(std::move(mimi_kv_tensor));
      step_inputs.push_back(View(&mimi_offset));
      step_inputs.push_back(View(&mimi_conv));
      step_inputs.push_back(View(&decode_steps));

      outputs = model_->RunStep(std::move(step_inputs));

      // Extract audio
      const float *audio_data = outputs[0].GetTensorData<float>();
      int32_t audio_len =
          outputs[0].GetTensorTypeAndShapeInfo().GetShape()[2];
      audio_samples.insert(audio_samples.end(), audio_data,
                           audio_data + audio_len);

      // Check EOS before moving outputs
      float eos_logit = outputs[2].GetTensorData<float>()[0];
      if (eos_frame < 0 && eos_logit > -1.0f) {
        eos_frame = frame;
        total_frames = eos_frame + frames_after_eos + 1;
        if (config_.model.debug) {
          SHERPA_ONNX_LOGE("EOS detected at frame %d, logit: %f", frame,
                           eos_logit);
        }
      }

      // Update latent, mimi_offset, mimi_conv
      latent = std::move(outputs[1]);
      mimi_offset = std::move(outputs[5]);
      mimi_conv = std::move(outputs[6]);

      // Update flow_kv cache: append flow_kv_new
      {
        const float *new_data = outputs[3].GetTensorData<float>();
        auto new_shape = outputs[3].GetTensorTypeAndShapeInfo().GetShape();
        size_t new_size = 1;
        for (auto d : new_shape) new_size *= d;
        flow_kv_cache.insert(flow_kv_cache.end(), new_data,
                             new_data + new_size);
      }

      // Update mimi_kv cache: append mimi_kv_new
      {
        const float *new_data = outputs[4].GetTensorData<float>();
        auto new_shape = outputs[4].GetTensorTypeAndShapeInfo().GetShape();
        size_t new_size = 1;
        for (auto d : new_shape) new_size *= d;
        mimi_kv_cache.insert(mimi_kv_cache.end(), new_data,
                             new_data + new_size);
      }

      is_bos = model_->CreateNonBosFlag();

      // Recreate latent_gates for next iteration
      latent_gates = model_->CreateLatentGates();

      if (eos_frame >= 0 && frame >= eos_frame + frames_after_eos) {
        if (config_.model.debug) {
          SHERPA_ONNX_LOGE("Stopping at frame %d (eos_frame=%d, "
                           "frames_after_eos=%d)",
                           frame, eos_frame, frames_after_eos);
        }
        break;
      }

      // Callback with progress
      if (callback) {
        float progress = static_cast<float>(frame + 1) / total_frames;
        if (progress > 1.0f) progress = 1.0f;
        should_continue = callback(audio_data, audio_len, progress);
      }
    }

    // Final callback to ensure100% is shown
    if (callback && should_continue) {
      callback(nullptr, 0, 1.0f);
    }

    GeneratedAudio ans;
    ans.sample_rate = SampleRate();
    ans.samples = std::move(audio_samples);

    return ans;
  }

  Ort::Value GetVoiceEmbedding(const GenerationConfig &gen_config) const {
    if (gen_config.reference_sample_rate <= 0) {
      SHERPA_ONNX_LOGE("reference_sample_rate %d is invalid.",
                       gen_config.reference_sample_rate);
      return Ort::Value{nullptr};
    }

    if (gen_config.reference_audio.empty()) {
      SHERPA_ONNX_LOGE("reference audio is empty");
      return Ort::Value{nullptr};
    }

    std::vector<float> reference_audio;

    const float *p_audio;
    int32_t num_samples;
    if (gen_config.reference_sample_rate != SampleRate()) {
      if (config_.model.debug) {
        SHERPA_ONNX_LOGE(
            "Creating a resampler:\n"
            "   in_sample_rate: %d\n"
            "   output_sample_rate: %d",
            gen_config.reference_sample_rate, SampleRate());
      }

      float min_freq =
          std::min<int32_t>(gen_config.reference_sample_rate, SampleRate());
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;

      int32_t lowpass_filter_width = 6;
      auto resampler = std::make_unique<sherpa_onnx::LinearResample>(
          gen_config.reference_sample_rate, SampleRate(), lowpass_cutoff,
          lowpass_filter_width);

      resampler->Resample(gen_config.reference_audio.data(),
                          gen_config.reference_audio.size(), true,
                          &reference_audio);
      p_audio = reference_audio.data();
      num_samples = reference_audio.size();
    } else {
      p_audio = gen_config.reference_audio.data();
      num_samples = gen_config.reference_audio.size();
    }

    float max_reference_audio_len =
        gen_config.GetExtraFloat("max_reference_audio_len", 10.0f);

    // in seconds
    int32_t max_len =
        static_cast<int32_t>(max_reference_audio_len * SampleRate());

    if (num_samples > max_len) {
      if (config_.model.debug) {
        SHERPA_ONNX_LOGE(
            "max_reference_audio_len is %.3f seconds. Given reference audio of "
            "%.3f seconds. Only the first %.3f seconds are used",
            max_reference_audio_len, num_samples * 1.0f / SampleRate(),
            max_reference_audio_len);
      }
      num_samples = max_len;
    }

    // Compute hash of reference audio for cache lookup
    size_t audio_hash = ComputeHash(p_audio, num_samples);

    auto cached_embedding = cache_.Get(audio_hash);
    if (cached_embedding) {
      if (config_.model.debug) {
        SHERPA_ONNX_LOGE("CACHE HIT: voice embedding (hash=%zu)", audio_hash);
      }
      // Create an owned tensor and copy data to avoid use-after-free
      auto result = Ort::Value::CreateTensor<float>(
          Ort::AllocatorWithDefaultOptions{}, cached_embedding->second.data(),
          cached_embedding->second.size());
      std::copy(cached_embedding->first.begin(), cached_embedding->first.end(),
                result.GetTensorMutableData<float>());
      return result;
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> shape = {1, 1, num_samples};
    Ort::Value x = Ort::Value::CreateTensor(memory_info,
                                             const_cast<float *>(p_audio),
                                             num_samples, shape.data(),
                                             shape.size());

    Ort::Value result = model_->RunEncoder(std::move(x));

    auto info = result.GetTensorTypeAndShapeInfo();
    auto result_shape = info.GetShape();
    size_t total = info.GetElementCount();
    const float *result_data = result.GetTensorData<float>();

    cache_.Put(audio_hash, std::vector<float>(result_data, result_data + total),
               std::move(result_shape));

    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("CACHE MISS: cached embedding (hash=%zu, %zu floats)",
                       audio_hash, total);
    }

    return result;
  }

  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsPocketZhEnModel> model_;
  std::unique_ptr<PocketZhEnLexicon> lexicon_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;

  // Shared Thread-Safe LRU Cache for Voice Embeddings
  struct VoiceEmbeddingCache {
    using Embedding = std::pair<std::vector<float>, std::vector<int64_t>>;
    using EmbeddingPtr = std::shared_ptr<Embedding>;

   private:
    using ListNode = std::pair<size_t, EmbeddingPtr>;
    using ListIt = std::list<ListNode>::iterator;

    mutable std::mutex mutex_;
    size_t capacity_;

    // Front = most recently used
    std::list<ListNode> lru_list_;

    // Key -> iterator into lru_list_
    std::unordered_map<size_t, ListIt> map_;

   public:
    static constexpr size_t kDefaultCapacity = 50;

    explicit VoiceEmbeddingCache(size_t cap = kDefaultCapacity)
        : capacity_(cap) {}

    EmbeddingPtr Get(size_t key) {
      std::lock_guard<std::mutex> lock(mutex_);

      auto it = map_.find(key);
      if (it == map_.end()) {
        return nullptr;  // cache miss
      }

      // Move to front (most recently used)
      if (it->second != lru_list_.begin()) {
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
      }

      return it->second->second;  // copy shared_ptr
    }

    void Put(size_t key, std::vector<float> data, std::vector<int64_t> shape) {
      std::lock_guard<std::mutex> lock(mutex_);

      if (capacity_ == 0) {
        return;
      }

      auto it = map_.find(key);

      // If exists, update and move to front
      if (it != map_.end()) {
        it->second->second =
            std::make_shared<Embedding>(std::move(data), std::move(shape));

        if (it->second != lru_list_.begin()) {
          lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        }
        return;
      }

      // Evict if full
      if (lru_list_.size() >= capacity_) {
        auto &last = lru_list_.back();
        size_t last_key = last.first;

        map_.erase(last_key);
        lru_list_.pop_back();  // shared_ptr released here
      }

      // Insert new at front
      lru_list_.emplace_front(
          key, std::make_shared<Embedding>(std::move(data), std::move(shape)));

      map_[key] = lru_list_.begin();
    }

    void SetCapacity(int32_t cap) {
      if (cap < 0) {
        SHERPA_ONNX_LOGE(
            "voice_embedding_cache_capacity must be >= 0. Given: %d", cap);
        SHERPA_ONNX_EXIT(-1);
      }

      std::lock_guard<std::mutex> lock(mutex_);
      capacity_ = cap;

      while (lru_list_.size() > capacity_) {
        auto &last = lru_list_.back();
        size_t last_key = last.first;

        map_.erase(last_key);
        lru_list_.pop_back();
      }
    }

    size_t Size() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return lru_list_.size();
    }

    void Clear() {
      std::lock_guard<std::mutex> lock(mutex_);
      map_.clear();
      lru_list_.clear();
    }
  };

  mutable VoiceEmbeddingCache cache_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_ZH_EN_IMPL_H_
