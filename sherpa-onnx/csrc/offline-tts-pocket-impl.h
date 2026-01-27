// sherpa-onnx/csrc/offline-tts-pocket-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_IMPL_H_

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <strstream>
#include <tuple>
#include <utility>
#include <vector>

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/normal-data-generator.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-pocket-model.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/sentence-piece-tokenizer.h"

namespace sherpa_onnx {

class OfflineTtsPocketImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsPocketImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsPocketModel>(config.model)) {
    InitTokenizer();

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
  OfflineTtsPocketImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsPocketModel>(mgr, config.model)) {
    InitTokenizer(mgr);

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
        std::istrstream is(buf.data(), buf.size());
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

        std::unique_ptr<std::istream> s(
            new std::istrstream(buf.data(), buf.size()));

        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(std::move(s)));

        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }  // for (; !reader->Done(); reader->Next())
      }  // for (const auto &f : files)
    }  // if (!config.rule_fars.empty())
  }

  int32_t SampleRate() const override { return 24000; }

  int32_t NumSpeakers() const override { return 1; }

  /**
   *
   * Supported extra parameters:
   *
   *  - max_frames, int, default 500
   *  - frames_after_eos, int, default 3
   *  - temperature, float, default 0.7
   *  - chunk_size, int, default 15
   *  - max_reference_audio_len, float, default 10, in seconds
   *  - max_char_in_sentence, int, default 200
   *  - min_char_in_sentence, int, default 30
   */
  GeneratedAudio Generate(
      const std::string &_text, const GenerationConfig &gen_config,
      GeneratedAudioCallback callback = nullptr) const override {
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("%s", gen_config.ToString().c_str());
    }

    std::string text = _text;
    if (config_.model.debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("Raw text: %{public}s", text.c_str());
#else
      SHERPA_ONNX_LOGE("Raw text: %s", text.c_str());
#endif
      std::ostringstream os;
      os << "In bytes (hex):\n";
      const auto p = reinterpret_cast<const uint8_t *>(text.c_str());
      for (int32_t i = 0; i != text.size(); ++i) {
        os << std::setw(2) << std::setfill('0') << std::hex
           << static_cast<uint32_t>(p[i]) << " ";
      }
      os << "\n";

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

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

    auto sentences = SplitByPunctuation(text);

    if (sentences.empty()) {
      return {};
    }

    int32_t max_char_in_sentence =
        gen_config.GetExtraInt("max_char_in_sentence", 200);

    int32_t min_char_in_sentence =
        gen_config.GetExtraInt("min_char_in_sentence", 30);

    sentences = MergeShortSentences(sentences, max_char_in_sentence);

    std::vector<std::string> final_chunks;
    for (const auto &s : sentences) {
      auto pieces = SplitLongSentence(s, max_char_in_sentence);
      final_chunks.insert(final_chunks.end(), pieces.begin(), pieces.end());
    }

    sentences = std::move(final_chunks);

    Ort::Value voice_embedding = GetVoiceEmbedding(gen_config);
    if (!voice_embedding) {
      return {};
    }

    GeneratedAudio result;
    result.sample_rate = SampleRate();

    const int32_t total = sentences.size();

    bool should_continue = true;

    for (int32_t i = 0; i < total && should_continue; ++i) {
      if (config_.model.debug) {
#if __OHOS__
        SHERPA_ONNX_LOGE("Processing %{public}d/%{public}d: %{public}s", i + 1,
                         total, sentences[i].c_str());
#else
        SHERPA_ONNX_LOGE("Processing %d/%d: %s", i + 1, total,
                         sentences[i].c_str());
#endif
      }
      GeneratedAudioCallback wrapped_cb = nullptr;

      if (callback) {
        wrapped_cb = [&, i](const float *samples, int32_t n,
                            float sentence_progress) -> bool {
          float global_progress = (i + sentence_progress) / total;

          return callback(samples, n, global_progress);
        };
      }

      GeneratedAudio cur = GenerateSingleSentence(sentences[i], gen_config,
                                                  View(&voice_embedding),
                                                  should_continue, wrapped_cb);

      if (cur.samples.empty()) {
        continue;
      }

      result.samples.insert(result.samples.end(), cur.samples.begin(),
                            cur.samples.end());
    }

    float silence_scale = gen_config.silence_scale;
    if (silence_scale != 1) {
      result = result.ScaleSilence(silence_scale);
    }

    return result;
  }

  static std::vector<std::string> MergeShortSentences(
      const std::vector<std::string> &sentences, size_t min_chars = 30) {
    std::vector<std::string> merged;
    std::string buffer;

    for (const auto &s : sentences) {
      if (!buffer.empty()) {
        buffer += " ";
      }
      buffer += s;

      if (buffer.size() >= min_chars) {
        merged.push_back(buffer);
        buffer.clear();
      }
    }

    if (!buffer.empty()) {
      merged.push_back(buffer);
    }

    return merged;
  }

  static std::vector<std::string> SplitLongSentence(const std::string &sentence,
                                                    size_t max_chars = 200) {
    std::vector<std::string> chunks;
    size_t start = 0;
    size_t len = sentence.size();

    while (start < len) {
      size_t end = start + max_chars;
      if (end >= len) {
        chunks.push_back(sentence.substr(start));
        break;
      }

      // Try to break at last space within max_chars
      size_t space_pos = sentence.rfind(' ', end);
      if (space_pos == std::string::npos || space_pos < start) {
        space_pos = end;  // no space found, force split
      }

      chunks.push_back(sentence.substr(start, space_pos - start));
      start = space_pos;

      // Skip spaces at start of next chunk
      while (start < len && sentence[start] == ' ') {
        ++start;
      }
    }

    return chunks;
  }

  static std::vector<std::string> SplitByPunctuation(const std::string &text) {
    std::vector<std::string> sentences;
    std::string cur;

    auto flush = [&]() {
      if (!cur.empty()) {
        // trim leading/trailing spaces
        auto begin = cur.find_first_not_of(" \t\n");
        auto end = cur.find_last_not_of(" \t\n");
        if (begin != std::string::npos) {
          sentences.emplace_back(cur.substr(begin, end - begin + 1));
        }
        cur.clear();
      }
    };

    for (char c : text) {
      cur.push_back(c);
      if (c == '.' || c == '!' || c == '?') {
        flush();
      }
    }

    flush();

    return sentences;
  }

  GeneratedAudio GenerateSingleSentence(
      const std::string &text, const GenerationConfig &gen_config,
      Ort::Value voice_embedding, bool &should_continue,
      GeneratedAudioCallback callback = nullptr) const {
    Ort::Value text_embedding = GetTextEmbedding(text);

    auto lm_main_state = model_->GetLmMainInitState();

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    {
      std::array<int64_t, 3> empty_seq_shape = {1, 0, 32};

      Ort::Value empty_seq_tensor = Ort::Value::CreateTensor<float>(
          memory_info, nullptr, 0, empty_seq_shape.data(),
          empty_seq_shape.size());

      // voice conditioning
      // discard the return result
      RunLmMain(View(&empty_seq_tensor), std::move(voice_embedding),
                lm_main_state);

      // text conditioning
      // discard the return result
      RunLmMain(std::move(empty_seq_tensor), std::move(text_embedding),
                lm_main_state);
    }

    std::vector<float> cur(1 * 1 * 32, std::numeric_limits<float>::quiet_NaN());
    std::array<int64_t, 3> cur_shape = {1, 1, 32};

    int32_t num_steps = gen_config.num_steps;
    int32_t max_frames = gen_config.GetExtraInt("max_frames", 500);
    int32_t frames_after_eos = gen_config.GetExtraInt("frames_after_eos", 3);
    float temperature = gen_config.GetExtraFloat("temperature", 0.7f);
    float stddev = std::sqrt(temperature);

    NormalDataGenerator normal_gen(0, stddev);
    std::vector<float> noise(32, 0);
    std::array<int64_t, 2> noise_shape = {1, 32};

    Ort::Value noise_tensor =
        Ort::Value::CreateTensor(memory_info, noise.data(), noise.size(),
                                 noise_shape.data(), noise_shape.size());

    std::array<int64_t, 3> empty_text_shape = {1, 0, 1024};

    Ort::Value empty_text_tensor = Ort::Value::CreateTensor<float>(
        memory_info, nullptr, 0, empty_text_shape.data(),
        empty_text_shape.size());

    Ort::Value conditioning;
    Ort::Value eos_logit;

    std::vector<float> latent_list;
    int32_t eos_step = -1;
    int32_t frame_size = -1;
    for (int32_t step = 0; step < max_frames; ++step) {
      Ort::Value cur_tensor =
          Ort::Value::CreateTensor(memory_info, cur.data(), cur.size(),
                                   cur_shape.data(), cur_shape.size());

      std::tie(conditioning, eos_logit) = RunLmMain(
          std::move(cur_tensor), View(&empty_text_tensor), lm_main_state);
      const float *p_logit = eos_logit.GetTensorData<float>();

      if (eos_step < 0 && p_logit[0] > -4) {
        eos_step = step;
      }

      if (eos_step > 0 && (step >= eos_step + frames_after_eos)) {
        break;
      }

      normal_gen.Fill(noise.data(), noise.size());

      Ort::Value latent =
          RunLmFlow(std::move(conditioning), View(&noise_tensor), num_steps);

      auto n = latent.GetTensorTypeAndShapeInfo().GetShape().back();
      if (frame_size == -1) {
        frame_size = n;
      }

      cur = {latent.GetTensorData<float>(), latent.GetTensorData<float>() + n};

      latent_list.insert(latent_list.end(), latent.GetTensorData<float>(),
                         latent.GetTensorData<float>() + n);
    }

    lm_main_state.values.clear();

    auto decoder_state = model_->GetMimiDecoderInitState();

    int32_t chunk_size = gen_config.GetExtraInt("chunk_size", 15);

    int32_t num_chunks = latent_list.size() / frame_size / chunk_size;
    std::array<int64_t, 3> chunk_shape = {1, chunk_size, frame_size};

    std::vector<float> audio_list;

    int32_t remaining_chunks =
        (latent_list.size() - num_chunks * chunk_size * frame_size) /
        frame_size;

    const float *p = latent_list.data();
    for (int32_t i = 0;
         (p < latent_list.data() + latent_list.size()) && should_continue;
         ++i) {
      int32_t this_chunk_size = chunk_size;
      if (i >= num_chunks) {
        this_chunk_size = remaining_chunks;
      }

      chunk_shape[1] = this_chunk_size;

      Ort::Value chunk_tensor = Ort::Value::CreateTensor(
          memory_info, const_cast<float *>(p), this_chunk_size * frame_size,
          chunk_shape.data(), chunk_shape.size());

      p += this_chunk_size * frame_size;

      Ort::Value out = RunMimiDecoder(std::move(chunk_tensor), decoder_state);

      auto n = out.GetTensorTypeAndShapeInfo().GetShape().back();

      if (callback) {
        should_continue =
            callback(out.GetTensorData<float>(), n,
                     (i + 1) * 1.0 / (num_chunks + !!remaining_chunks));
        // Caution(fangjun): out is freed when the callback returns, so users
        // should copy the data if they want to access the data after
        // the callback returns to avoid segmentation fault.
      }

      audio_list.insert(audio_list.end(), out.GetTensorData<float>(),
                        out.GetTensorData<float>() + n);
    }

    GeneratedAudio ans;
    ans.sample_rate = SampleRate();
    ans.samples = std::move(audio_list);

    return ans;
  }

 private:
  template <typename Manager>
  void InitTokenizer(Manager *mgr) {
    tokenizer_ = std::make_unique<SentencePieceTokenizer>(
        mgr, config_.model.pocket.vocab_json,
        config_.model.pocket.token_scores_json);
  }

  void InitTokenizer() {
    tokenizer_ = std::make_unique<SentencePieceTokenizer>(
        config_.model.pocket.vocab_json,
        config_.model.pocket.token_scores_json);
  }

  Ort::Value GetVoiceEmbedding(const GenerationConfig &gen_config) const {
    if (gen_config.reference_sample_rate <= 0) {
      SHERPA_ONNX_LOGE("reference_sample_rate %d is invalid.",
                       gen_config.reference_sample_rate);
      return nullptr;
    }

    if (gen_config.reference_audio.empty()) {
      SHERPA_ONNX_LOGE("reference audio is empty");
      return nullptr;
    }

    std::vector<float> reference_audio;

    const float *p_audio;
    int32_t num_samples;
    if (gen_config.reference_sample_rate != SampleRate()) {
      SHERPA_ONNX_LOGE(
          "Creating a resampler:\n"
          "   in_sample_rate: %d\n"
          "   output_sample_rate: %d",
          gen_config.reference_sample_rate, SampleRate());

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

    // in seconds
    float max_reference_audio_len =
        gen_config.GetExtraFloat("max_reference_audio_len", 10);

    int32_t max_len =
        static_cast<int32_t>(max_reference_audio_len * SampleRate());

    if (num_samples > max_len) {
      SHERPA_ONNX_LOGE(
          "max_reference_audio_len is %.3f seconds. Given reference audio of "
          "%.3f seconds. Only the first %.3f seconds are used",
          max_reference_audio_len, num_samples * 1.0f / SampleRate(),
          max_reference_audio_len);
      num_samples = max_len;
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> shape = {1, 1, num_samples};
    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, const_cast<float *>(p_audio),
                                 num_samples, shape.data(), shape.size());
    return model_->RunMimiEncoder(std::move(x));
  }

  Ort::Value GetTextEmbedding(const std::string &text) const {
    std::vector<int32_t> token_ids = tokenizer_->EncodeIds(text);
    if (config_.model.debug) {
      std::ostringstream os;
      os << "\ntoken_ids (len=" << token_ids.size() << "): ";
      for (auto i : token_ids) {
        os << i << " ";
      }
      os << "\n";

      auto tokens = tokenizer_->EncodeTokens(text);
      os << "tokens (len=" << tokens.size() << "):";
      for (const auto &t : tokens) {
        os << t << " ";
      }

      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }

    std::vector<int64_t> token_ids_i64 = {token_ids.begin(), token_ids.end()};

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> shape = {1,
                                    static_cast<int64_t>(token_ids_i64.size())};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, token_ids_i64.data(),
                                            token_ids_i64.size(), shape.data(),
                                            shape.size());
    return model_->RunTextConditioner(std::move(x));
  }

  // state is changed in-place
  std::pair<Ort::Value, Ort::Value> RunLmMain(Ort::Value seq,
                                              Ort::Value embedding,
                                              PocketLmMainState &state) const {
    std::tuple<Ort::Value, Ort::Value, PocketLmMainState> output =
        model_->RunLmMain(std::move(seq), std::move(embedding),
                          std::move(state));

    state = std::move(std::get<2>(output));

    return {std::move(std::get<0>(output)), std::move(std::get<1>(output))};
  }

  Ort::Value RunLmFlow(Ort::Value conditioning, Ort::Value noise,
                       int32_t num_steps) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value x = Clone(model_->Allocator(), &noise);

    std::array<int64_t, 2> shape = {1, 1};

    float dt = 1.0f / static_cast<float>(num_steps);

    float s = 0;
    float t = 0;

    Ort::Value s_tensor = Ort::Value::CreateTensor(memory_info, &s, 1,
                                                   shape.data(), shape.size());

    Ort::Value t_tensor = Ort::Value::CreateTensor(memory_info, &t, 1,
                                                   shape.data(), shape.size());

    for (int32_t i = 0; i < num_steps; ++i) {
      s = static_cast<float>(i) / static_cast<float>(num_steps);
      t = s + dt;

      Ort::Value out = model_->RunLmFlow(View(&conditioning), View(&s_tensor),
                                         View(&t_tensor), View(&x));

      auto n = out.GetTensorTypeAndShapeInfo().GetShape().back();

      ScaleAdd(out.GetTensorData<float>(), dt, n,
               x.GetTensorMutableData<float>());
    }

    return std::move(x);
  }

  // state is changed in-place
  Ort::Value RunMimiDecoder(Ort::Value latent,
                            PocketMimiDecoderState &state) const {
    std::pair<Ort::Value, PocketMimiDecoderState> output =
        model_->RunMimiDecoder(std::move(latent), std::move(state));

    state = std::move(output.second);

    return std::move(output.first);
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsPocketModel> model_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
  std::unique_ptr<SentencePieceTokenizer> tokenizer_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_IMPL_H_
