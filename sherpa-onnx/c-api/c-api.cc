// sherpa-onnx/c-api/c-api.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/c-api/c-api.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <strstream>
#include <utility>
#include <vector>

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/audio-tagging.h"
#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-punctuation.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/csrc/online-punctuation.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"
#include "sherpa-onnx/csrc/speaker-embedding-manager.h"
#include "sherpa-onnx/csrc/spoken-language-identification.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/version.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

#if SHERPA_ONNX_ENABLE_TTS == 1
#include "sherpa-onnx/csrc/offline-tts.h"
#endif

#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1
#include "sherpa-onnx/csrc/offline-speaker-diarization.h"
#endif

const char *SherpaOnnxGetVersionStr() { return sherpa_onnx::GetVersionStr(); }
const char *SherpaOnnxGetGitSha1() { return sherpa_onnx::GetGitSha1(); }
const char *SherpaOnnxGetGitDate() { return sherpa_onnx::GetGitDate(); }

struct SherpaOnnxOnlineRecognizer {
  std::unique_ptr<sherpa_onnx::OnlineRecognizer> impl;
};

struct SherpaOnnxOnlineStream {
  std::unique_ptr<sherpa_onnx::OnlineStream> impl;
  explicit SherpaOnnxOnlineStream(std::unique_ptr<sherpa_onnx::OnlineStream> p)
      : impl(std::move(p)) {}
};

struct SherpaOnnxDisplay {
  std::unique_ptr<sherpa_onnx::Display> impl;
};

#define SHERPA_ONNX_OR(x, y) (x ? x : y)

static sherpa_onnx::OnlineRecognizerConfig GetOnlineRecognizerConfig(
    const SherpaOnnxOnlineRecognizerConfig *config) {
  sherpa_onnx::OnlineRecognizerConfig recognizer_config;

  recognizer_config.feat_config.sampling_rate =
      SHERPA_ONNX_OR(config->feat_config.sample_rate, 16000);
  recognizer_config.feat_config.feature_dim =
      SHERPA_ONNX_OR(config->feat_config.feature_dim, 80);

  recognizer_config.model_config.transducer.encoder =
      SHERPA_ONNX_OR(config->model_config.transducer.encoder, "");
  recognizer_config.model_config.transducer.decoder =
      SHERPA_ONNX_OR(config->model_config.transducer.decoder, "");
  recognizer_config.model_config.transducer.joiner =
      SHERPA_ONNX_OR(config->model_config.transducer.joiner, "");

  recognizer_config.model_config.paraformer.encoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.encoder, "");
  recognizer_config.model_config.paraformer.decoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.decoder, "");

  recognizer_config.model_config.zipformer2_ctc.model =
      SHERPA_ONNX_OR(config->model_config.zipformer2_ctc.model, "");

  recognizer_config.model_config.tokens =
      SHERPA_ONNX_OR(config->model_config.tokens, "");
  if (config->model_config.tokens_buf &&
      config->model_config.tokens_buf_size > 0) {
    recognizer_config.model_config.tokens_buf = std::string(
        config->model_config.tokens_buf, config->model_config.tokens_buf_size);
  }

  recognizer_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  recognizer_config.model_config.provider_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");

  if (recognizer_config.model_config.provider_config.provider.empty()) {
    recognizer_config.model_config.provider_config.provider = "cpu";
  }

  recognizer_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");
  recognizer_config.model_config.debug =
      SHERPA_ONNX_OR(config->model_config.debug, 0);
  recognizer_config.model_config.modeling_unit =
      SHERPA_ONNX_OR(config->model_config.modeling_unit, "cjkchar");

  if (recognizer_config.model_config.modeling_unit.empty()) {
    recognizer_config.model_config.modeling_unit = "cjkchar";
  }

  recognizer_config.model_config.bpe_vocab =
      SHERPA_ONNX_OR(config->model_config.bpe_vocab, "");

  recognizer_config.decoding_method =
      SHERPA_ONNX_OR(config->decoding_method, "greedy_search");
  if (recognizer_config.decoding_method.empty()) {
    recognizer_config.decoding_method = "greedy_search";
  }

  recognizer_config.max_active_paths =
      SHERPA_ONNX_OR(config->max_active_paths, 4);

  recognizer_config.enable_endpoint =
      SHERPA_ONNX_OR(config->enable_endpoint, 0);

  recognizer_config.endpoint_config.rule1.min_trailing_silence =
      SHERPA_ONNX_OR(config->rule1_min_trailing_silence, 2.4);

  recognizer_config.endpoint_config.rule2.min_trailing_silence =
      SHERPA_ONNX_OR(config->rule2_min_trailing_silence, 1.2);

  recognizer_config.endpoint_config.rule3.min_utterance_length =
      SHERPA_ONNX_OR(config->rule3_min_utterance_length, 20);

  recognizer_config.hotwords_file = SHERPA_ONNX_OR(config->hotwords_file, "");
  recognizer_config.hotwords_score =
      SHERPA_ONNX_OR(config->hotwords_score, 1.5);
  if (config->hotwords_buf && config->hotwords_buf_size > 0) {
    recognizer_config.hotwords_buf =
        std::string(config->hotwords_buf, config->hotwords_buf_size);
  }

  recognizer_config.blank_penalty = config->blank_penalty;

  recognizer_config.ctc_fst_decoder_config.graph =
      SHERPA_ONNX_OR(config->ctc_fst_decoder_config.graph, "");
  recognizer_config.ctc_fst_decoder_config.max_active =
      SHERPA_ONNX_OR(config->ctc_fst_decoder_config.max_active, 3000);

  recognizer_config.rule_fsts = SHERPA_ONNX_OR(config->rule_fsts, "");
  recognizer_config.rule_fars = SHERPA_ONNX_OR(config->rule_fars, "");

  recognizer_config.hr.dict_dir = SHERPA_ONNX_OR(config->hr.dict_dir, "");
  recognizer_config.hr.lexicon = SHERPA_ONNX_OR(config->hr.lexicon, "");
  recognizer_config.hr.rule_fsts = SHERPA_ONNX_OR(config->hr.rule_fsts, "");

  if (config->model_config.debug) {
#if __OHOS__
    auto str_vec = sherpa_onnx::SplitString(recognizer_config.ToString(), 128);
    for (const auto &s : str_vec) {
      SHERPA_ONNX_LOGE("%{public}s\n", s.c_str());
      SHERPA_ONNX_LOGE("%s\n", s.c_str());
    }
#else
    SHERPA_ONNX_LOGE("%s", recognizer_config.ToString().c_str());
#endif
  }

  return recognizer_config;
}

const SherpaOnnxOnlineRecognizer *SherpaOnnxCreateOnlineRecognizer(
    const SherpaOnnxOnlineRecognizerConfig *config) {
  sherpa_onnx::OnlineRecognizerConfig recognizer_config =
      GetOnlineRecognizerConfig(config);

  if (!recognizer_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config!");
    return nullptr;
  }

  SherpaOnnxOnlineRecognizer *recognizer = new SherpaOnnxOnlineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_onnx::OnlineRecognizer>(recognizer_config);

  return recognizer;
}

void SherpaOnnxDestroyOnlineRecognizer(
    const SherpaOnnxOnlineRecognizer *recognizer) {
  delete recognizer;
}

const SherpaOnnxOnlineStream *SherpaOnnxCreateOnlineStream(
    const SherpaOnnxOnlineRecognizer *recognizer) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(recognizer->impl->CreateStream());
  return stream;
}

const SherpaOnnxOnlineStream *SherpaOnnxCreateOnlineStreamWithHotwords(
    const SherpaOnnxOnlineRecognizer *recognizer, const char *hotwords) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(recognizer->impl->CreateStream(hotwords));
  return stream;
}

void SherpaOnnxDestroyOnlineStream(const SherpaOnnxOnlineStream *stream) {
  delete stream;
}

void SherpaOnnxOnlineStreamAcceptWaveform(const SherpaOnnxOnlineStream *stream,
                                          int32_t sample_rate,
                                          const float *samples, int32_t n) {
  stream->impl->AcceptWaveform(sample_rate, samples, n);
}

int32_t SherpaOnnxIsOnlineStreamReady(
    const SherpaOnnxOnlineRecognizer *recognizer,
    const SherpaOnnxOnlineStream *stream) {
  return recognizer->impl->IsReady(stream->impl.get());
}

void SherpaOnnxDecodeOnlineStream(const SherpaOnnxOnlineRecognizer *recognizer,
                                  const SherpaOnnxOnlineStream *stream) {
  recognizer->impl->DecodeStream(stream->impl.get());
}

void SherpaOnnxDecodeMultipleOnlineStreams(
    const SherpaOnnxOnlineRecognizer *recognizer,
    const SherpaOnnxOnlineStream **streams, int32_t n) {
  std::vector<sherpa_onnx::OnlineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  recognizer->impl->DecodeStreams(ss.data(), n);
}

const SherpaOnnxOnlineRecognizerResult *SherpaOnnxGetOnlineStreamResult(
    const SherpaOnnxOnlineRecognizer *recognizer,
    const SherpaOnnxOnlineStream *stream) {
  sherpa_onnx::OnlineRecognizerResult result =
      recognizer->impl->GetResult(stream->impl.get());
  const auto &text = result.text;

  auto r = new SherpaOnnxOnlineRecognizerResult;
  memset(r, 0, sizeof(SherpaOnnxOnlineRecognizerResult));

  // copy text
  char *pText = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), pText);
  pText[text.size()] = 0;
  r->text = pText;

  // copy json
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  r->json = pJson;

  // copy tokens
  auto count = result.tokens.size();
  if (count > 0) {
    size_t total_length = 0;
    for (const auto &token : result.tokens) {
      // +1 for the null character at the end of each token
      total_length += token.size() + 1;
    }

    r->count = count;
    // Each word ends with nullptr
    char *tokens = new char[total_length]{};
    char **tokens_temp = new char *[r->count];
    int32_t pos = 0;
    for (int32_t i = 0; i < r->count; ++i) {
      tokens_temp[i] = tokens + pos;
      memcpy(tokens + pos, result.tokens[i].c_str(), result.tokens[i].size());
      // +1 to move past the null character
      pos += result.tokens[i].size() + 1;
    }
    r->tokens_arr = tokens_temp;

    if (!result.timestamps.empty() && result.timestamps.size() == r->count) {
      r->timestamps = new float[r->count];
      std::copy(result.timestamps.begin(), result.timestamps.end(),
                r->timestamps);
    } else {
      r->timestamps = nullptr;
    }

    r->tokens = tokens;
  } else {
    r->count = 0;
    r->timestamps = nullptr;
    r->tokens = nullptr;
    r->tokens_arr = nullptr;
  }

  return r;
}

void SherpaOnnxDestroyOnlineRecognizerResult(
    const SherpaOnnxOnlineRecognizerResult *r) {
  if (r) {
    delete[] r->text;
    delete[] r->json;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->timestamps;
    delete r;
  }
}

const char *SherpaOnnxGetOnlineStreamResultAsJson(
    const SherpaOnnxOnlineRecognizer *recognizer,
    const SherpaOnnxOnlineStream *stream) {
  sherpa_onnx::OnlineRecognizerResult result =
      recognizer->impl->GetResult(stream->impl.get());
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  return pJson;
}

void SherpaOnnxDestroyOnlineStreamResultJson(const char *s) { delete[] s; }

void SherpaOnnxOnlineStreamReset(const SherpaOnnxOnlineRecognizer *recognizer,
                                 const SherpaOnnxOnlineStream *stream) {
  recognizer->impl->Reset(stream->impl.get());
}

void SherpaOnnxOnlineStreamInputFinished(const SherpaOnnxOnlineStream *stream) {
  stream->impl->InputFinished();
}

int32_t SherpaOnnxOnlineStreamIsEndpoint(
    const SherpaOnnxOnlineRecognizer *recognizer,
    const SherpaOnnxOnlineStream *stream) {
  return recognizer->impl->IsEndpoint(stream->impl.get());
}

const SherpaOnnxDisplay *SherpaOnnxCreateDisplay(int32_t max_word_per_line) {
  SherpaOnnxDisplay *ans = new SherpaOnnxDisplay;
  ans->impl = std::make_unique<sherpa_onnx::Display>(max_word_per_line);
  return ans;
}

void SherpaOnnxDestroyDisplay(const SherpaOnnxDisplay *display) {
  delete display;
}

void SherpaOnnxPrint(const SherpaOnnxDisplay *display, int32_t idx,
                     const char *s) {
  display->impl->Print(idx, s);
}

// ============================================================
// For offline ASR (i.e., non-streaming ASR)
// ============================================================
//
struct SherpaOnnxOfflineRecognizer {
  std::unique_ptr<sherpa_onnx::OfflineRecognizer> impl;
};

struct SherpaOnnxOfflineStream {
  std::unique_ptr<sherpa_onnx::OfflineStream> impl;
  explicit SherpaOnnxOfflineStream(
      std::unique_ptr<sherpa_onnx::OfflineStream> p)
      : impl(std::move(p)) {}
};

static sherpa_onnx::OfflineRecognizerConfig GetOfflineRecognizerConfig(
    const SherpaOnnxOfflineRecognizerConfig *config) {
  sherpa_onnx::OfflineRecognizerConfig recognizer_config;

  recognizer_config.feat_config.sampling_rate =
      SHERPA_ONNX_OR(config->feat_config.sample_rate, 16000);

  recognizer_config.feat_config.feature_dim =
      SHERPA_ONNX_OR(config->feat_config.feature_dim, 80);

  recognizer_config.model_config.transducer.encoder_filename =
      SHERPA_ONNX_OR(config->model_config.transducer.encoder, "");

  recognizer_config.model_config.transducer.decoder_filename =
      SHERPA_ONNX_OR(config->model_config.transducer.decoder, "");

  recognizer_config.model_config.transducer.joiner_filename =
      SHERPA_ONNX_OR(config->model_config.transducer.joiner, "");

  recognizer_config.model_config.paraformer.model =
      SHERPA_ONNX_OR(config->model_config.paraformer.model, "");

  recognizer_config.model_config.nemo_ctc.model =
      SHERPA_ONNX_OR(config->model_config.nemo_ctc.model, "");

  recognizer_config.model_config.whisper.encoder =
      SHERPA_ONNX_OR(config->model_config.whisper.encoder, "");

  recognizer_config.model_config.whisper.decoder =
      SHERPA_ONNX_OR(config->model_config.whisper.decoder, "");

  recognizer_config.model_config.whisper.language =
      SHERPA_ONNX_OR(config->model_config.whisper.language, "");

  recognizer_config.model_config.whisper.task =
      SHERPA_ONNX_OR(config->model_config.whisper.task, "transcribe");
  if (recognizer_config.model_config.whisper.task.empty()) {
    recognizer_config.model_config.whisper.task = "transcribe";
  }

  recognizer_config.model_config.whisper.tail_paddings =
      SHERPA_ONNX_OR(config->model_config.whisper.tail_paddings, -1);

  recognizer_config.model_config.tdnn.model =
      SHERPA_ONNX_OR(config->model_config.tdnn.model, "");

  recognizer_config.model_config.tokens =
      SHERPA_ONNX_OR(config->model_config.tokens, "");
  recognizer_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  recognizer_config.model_config.debug =
      SHERPA_ONNX_OR(config->model_config.debug, 0);
  recognizer_config.model_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");
  if (recognizer_config.model_config.provider.empty()) {
    recognizer_config.model_config.provider = "cpu";
  }

  recognizer_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");
  recognizer_config.model_config.modeling_unit =
      SHERPA_ONNX_OR(config->model_config.modeling_unit, "cjkchar");

  if (recognizer_config.model_config.modeling_unit.empty()) {
    recognizer_config.model_config.modeling_unit = "cjkchar";
  }

  recognizer_config.model_config.bpe_vocab =
      SHERPA_ONNX_OR(config->model_config.bpe_vocab, "");

  recognizer_config.model_config.telespeech_ctc =
      SHERPA_ONNX_OR(config->model_config.telespeech_ctc, "");

  recognizer_config.model_config.sense_voice.model =
      SHERPA_ONNX_OR(config->model_config.sense_voice.model, "");

  recognizer_config.model_config.sense_voice.language =
      SHERPA_ONNX_OR(config->model_config.sense_voice.language, "");

  recognizer_config.model_config.sense_voice.use_itn =
      config->model_config.sense_voice.use_itn;

  recognizer_config.model_config.moonshine.preprocessor =
      SHERPA_ONNX_OR(config->model_config.moonshine.preprocessor, "");

  recognizer_config.model_config.moonshine.encoder =
      SHERPA_ONNX_OR(config->model_config.moonshine.encoder, "");

  recognizer_config.model_config.moonshine.uncached_decoder =
      SHERPA_ONNX_OR(config->model_config.moonshine.uncached_decoder, "");

  recognizer_config.model_config.moonshine.cached_decoder =
      SHERPA_ONNX_OR(config->model_config.moonshine.cached_decoder, "");

  recognizer_config.model_config.fire_red_asr.encoder =
      SHERPA_ONNX_OR(config->model_config.fire_red_asr.encoder, "");

  recognizer_config.model_config.fire_red_asr.decoder =
      SHERPA_ONNX_OR(config->model_config.fire_red_asr.decoder, "");

  recognizer_config.model_config.dolphin.model =
      SHERPA_ONNX_OR(config->model_config.dolphin.model, "");

  recognizer_config.model_config.zipformer_ctc.model =
      SHERPA_ONNX_OR(config->model_config.zipformer_ctc.model, "");

  recognizer_config.model_config.canary.encoder =
      SHERPA_ONNX_OR(config->model_config.canary.encoder, "");

  recognizer_config.model_config.canary.decoder =
      SHERPA_ONNX_OR(config->model_config.canary.decoder, "");

  recognizer_config.model_config.canary.src_lang =
      SHERPA_ONNX_OR(config->model_config.canary.src_lang, "");

  recognizer_config.model_config.canary.tgt_lang =
      SHERPA_ONNX_OR(config->model_config.canary.tgt_lang, "");

  recognizer_config.model_config.canary.use_pnc =
      config->model_config.canary.use_pnc;

  recognizer_config.lm_config.model =
      SHERPA_ONNX_OR(config->lm_config.model, "");
  recognizer_config.lm_config.scale =
      SHERPA_ONNX_OR(config->lm_config.scale, 1.0);

  recognizer_config.decoding_method =
      SHERPA_ONNX_OR(config->decoding_method, "greedy_search");

  if (recognizer_config.decoding_method.empty()) {
    recognizer_config.decoding_method = "greedy_search";
  }

  recognizer_config.max_active_paths =
      SHERPA_ONNX_OR(config->max_active_paths, 4);

  recognizer_config.hotwords_file = SHERPA_ONNX_OR(config->hotwords_file, "");
  recognizer_config.hotwords_score =
      SHERPA_ONNX_OR(config->hotwords_score, 1.5);

  recognizer_config.blank_penalty = config->blank_penalty;

  recognizer_config.rule_fsts = SHERPA_ONNX_OR(config->rule_fsts, "");
  recognizer_config.rule_fars = SHERPA_ONNX_OR(config->rule_fars, "");

  recognizer_config.hr.dict_dir = SHERPA_ONNX_OR(config->hr.dict_dir, "");
  recognizer_config.hr.lexicon = SHERPA_ONNX_OR(config->hr.lexicon, "");
  recognizer_config.hr.rule_fsts = SHERPA_ONNX_OR(config->hr.rule_fsts, "");

  if (config->model_config.debug) {
#if __OHOS__
    auto str_vec = sherpa_onnx::SplitString(recognizer_config.ToString(), 128);
    for (const auto &s : str_vec) {
      SHERPA_ONNX_LOGE("%{public}s\n", s.c_str());
      SHERPA_ONNX_LOGE("%s\n", s.c_str());
    }
#else
    SHERPA_ONNX_LOGE("%s", recognizer_config.ToString().c_str());
#endif
  }

  return recognizer_config;
}

const SherpaOnnxOfflineRecognizer *SherpaOnnxCreateOfflineRecognizer(
    const SherpaOnnxOfflineRecognizerConfig *config) {
  sherpa_onnx::OfflineRecognizerConfig recognizer_config =
      GetOfflineRecognizerConfig(config);

  if (!recognizer_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxOfflineRecognizer *recognizer = new SherpaOnnxOfflineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_onnx::OfflineRecognizer>(recognizer_config);

  return recognizer;
}

void SherpaOnnxOfflineRecognizerSetConfig(
    const SherpaOnnxOfflineRecognizer *recognizer,
    const SherpaOnnxOfflineRecognizerConfig *config) {
  sherpa_onnx::OfflineRecognizerConfig recognizer_config =
      GetOfflineRecognizerConfig(config);
  recognizer->impl->SetConfig(recognizer_config);
}

void SherpaOnnxDestroyOfflineRecognizer(
    const SherpaOnnxOfflineRecognizer *recognizer) {
  delete recognizer;
}

const SherpaOnnxOfflineStream *SherpaOnnxCreateOfflineStream(
    const SherpaOnnxOfflineRecognizer *recognizer) {
  SherpaOnnxOfflineStream *stream =
      new SherpaOnnxOfflineStream(recognizer->impl->CreateStream());
  return stream;
}

const SherpaOnnxOfflineStream *SherpaOnnxCreateOfflineStreamWithHotwords(
    const SherpaOnnxOfflineRecognizer *recognizer, const char *hotwords) {
  SherpaOnnxOfflineStream *stream =
      new SherpaOnnxOfflineStream(recognizer->impl->CreateStream(hotwords));
  return stream;
}

void SherpaOnnxDestroyOfflineStream(const SherpaOnnxOfflineStream *stream) {
  delete stream;
}

void SherpaOnnxAcceptWaveformOffline(const SherpaOnnxOfflineStream *stream,
                                     int32_t sample_rate, const float *samples,
                                     int32_t n) {
  stream->impl->AcceptWaveform(sample_rate, samples, n);
}

void SherpaOnnxDecodeOfflineStream(
    const SherpaOnnxOfflineRecognizer *recognizer,
    const SherpaOnnxOfflineStream *stream) {
  recognizer->impl->DecodeStream(stream->impl.get());
}

void SherpaOnnxDecodeMultipleOfflineStreams(
    const SherpaOnnxOfflineRecognizer *recognizer,
    const SherpaOnnxOfflineStream **streams, int32_t n) {
  std::vector<sherpa_onnx::OfflineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  recognizer->impl->DecodeStreams(ss.data(), n);
}

const SherpaOnnxOfflineRecognizerResult *SherpaOnnxGetOfflineStreamResult(
    const SherpaOnnxOfflineStream *stream) {
  const sherpa_onnx::OfflineRecognitionResult &result =
      stream->impl->GetResult();
  const auto &text = result.text;

  auto r = new SherpaOnnxOfflineRecognizerResult;
  memset(r, 0, sizeof(SherpaOnnxOfflineRecognizerResult));

  char *pText = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), pText);
  pText[text.size()] = 0;
  r->text = pText;

  // lang
  const auto &lang = result.lang;
  char *c_lang = new char[lang.size() + 1];
  std::copy(lang.begin(), lang.end(), c_lang);
  c_lang[lang.size()] = '\0';
  r->lang = c_lang;

  // emotion
  const auto &emotion = result.emotion;
  char *c_emotion = new char[emotion.size() + 1];
  std::copy(emotion.begin(), emotion.end(), c_emotion);
  c_emotion[emotion.size()] = '\0';
  r->emotion = c_emotion;

  // event
  const auto &event = result.event;
  char *c_event = new char[event.size() + 1];
  std::copy(event.begin(), event.end(), c_event);
  c_event[event.size()] = '\0';
  r->event = c_event;

  // copy json
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  r->json = pJson;

  // copy tokens
  auto count = result.tokens.size();
  if (count > 0) {
    size_t total_length = 0;
    for (const auto &token : result.tokens) {
      // +1 for the null character at the end of each token
      total_length += token.size() + 1;
    }

    r->count = count;
    // Each word ends with nullptr
    char *tokens = new char[total_length]{};
    char **tokens_temp = new char *[r->count];
    int32_t pos = 0;
    for (int32_t i = 0; i < r->count; ++i) {
      tokens_temp[i] = tokens + pos;
      memcpy(tokens + pos, result.tokens[i].c_str(), result.tokens[i].size());
      // +1 to move past the null character
      pos += result.tokens[i].size() + 1;
    }
    r->tokens_arr = tokens_temp;

    if (!result.timestamps.empty() && result.timestamps.size() == r->count) {
      r->timestamps = new float[r->count];
      std::copy(result.timestamps.begin(), result.timestamps.end(),
                r->timestamps);
    } else {
      r->timestamps = nullptr;
    }

    r->tokens = tokens;
  } else {
    r->count = 0;
    r->timestamps = nullptr;
    r->tokens = nullptr;
    r->tokens_arr = nullptr;
  }

  return r;
}

void SherpaOnnxDestroyOfflineRecognizerResult(
    const SherpaOnnxOfflineRecognizerResult *r) {
  if (r) {
    delete[] r->text;
    delete[] r->timestamps;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->json;
    delete[] r->lang;
    delete[] r->emotion;
    delete[] r->event;
    delete r;
  }
}

const char *SherpaOnnxGetOfflineStreamResultAsJson(
    const SherpaOnnxOfflineStream *stream) {
  const sherpa_onnx::OfflineRecognitionResult &result =
      stream->impl->GetResult();
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  return pJson;
}

void SherpaOnnxDestroyOfflineStreamResultJson(const char *s) { delete[] s; }

// ============================================================
// For Keyword Spot
// ============================================================

struct SherpaOnnxKeywordSpotter {
  std::unique_ptr<sherpa_onnx::KeywordSpotter> impl;
};

static sherpa_onnx::KeywordSpotterConfig GetKeywordSpotterConfig(
    const SherpaOnnxKeywordSpotterConfig *config) {
  sherpa_onnx::KeywordSpotterConfig spotter_config;

  spotter_config.feat_config.sampling_rate =
      SHERPA_ONNX_OR(config->feat_config.sample_rate, 16000);
  spotter_config.feat_config.feature_dim =
      SHERPA_ONNX_OR(config->feat_config.feature_dim, 80);

  spotter_config.model_config.transducer.encoder =
      SHERPA_ONNX_OR(config->model_config.transducer.encoder, "");
  spotter_config.model_config.transducer.decoder =
      SHERPA_ONNX_OR(config->model_config.transducer.decoder, "");
  spotter_config.model_config.transducer.joiner =
      SHERPA_ONNX_OR(config->model_config.transducer.joiner, "");

  spotter_config.model_config.paraformer.encoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.encoder, "");
  spotter_config.model_config.paraformer.decoder =
      SHERPA_ONNX_OR(config->model_config.paraformer.decoder, "");

  spotter_config.model_config.zipformer2_ctc.model =
      SHERPA_ONNX_OR(config->model_config.zipformer2_ctc.model, "");

  spotter_config.model_config.tokens =
      SHERPA_ONNX_OR(config->model_config.tokens, "");
  if (config->model_config.tokens_buf &&
      config->model_config.tokens_buf_size > 0) {
    spotter_config.model_config.tokens_buf = std::string(
        config->model_config.tokens_buf, config->model_config.tokens_buf_size);
  }

  spotter_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  spotter_config.model_config.provider_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");
  if (spotter_config.model_config.provider_config.provider.empty()) {
    spotter_config.model_config.provider_config.provider = "cpu";
  }

  spotter_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");
  spotter_config.model_config.debug =
      SHERPA_ONNX_OR(config->model_config.debug, 0);

  spotter_config.max_active_paths = SHERPA_ONNX_OR(config->max_active_paths, 4);

  spotter_config.num_trailing_blanks =
      SHERPA_ONNX_OR(config->num_trailing_blanks, 1);

  spotter_config.keywords_score = SHERPA_ONNX_OR(config->keywords_score, 1.0);

  spotter_config.keywords_threshold =
      SHERPA_ONNX_OR(config->keywords_threshold, 0.25);

  spotter_config.keywords_file = SHERPA_ONNX_OR(config->keywords_file, "");
  if (config->keywords_buf && config->keywords_buf_size > 0) {
    spotter_config.keywords_buf =
        std::string(config->keywords_buf, config->keywords_buf_size);
  }

  if (spotter_config.model_config.debug) {
#if OHOS
    SHERPA_ONNX_LOGE("%{public}s\n", spotter_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", spotter_config.ToString().c_str());
#endif
  }

  return spotter_config;
}

const SherpaOnnxKeywordSpotter *SherpaOnnxCreateKeywordSpotter(
    const SherpaOnnxKeywordSpotterConfig *config) {
  auto spotter_config = GetKeywordSpotterConfig(config);
  if (!spotter_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config!");
    return nullptr;
  }

  SherpaOnnxKeywordSpotter *spotter = new SherpaOnnxKeywordSpotter;

  spotter->impl = std::make_unique<sherpa_onnx::KeywordSpotter>(spotter_config);

  return spotter;
}

void SherpaOnnxDestroyKeywordSpotter(const SherpaOnnxKeywordSpotter *spotter) {
  delete spotter;
}

const SherpaOnnxOnlineStream *SherpaOnnxCreateKeywordStream(
    const SherpaOnnxKeywordSpotter *spotter) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(spotter->impl->CreateStream());
  return stream;
}

const SherpaOnnxOnlineStream *SherpaOnnxCreateKeywordStreamWithKeywords(
    const SherpaOnnxKeywordSpotter *spotter, const char *keywords) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(spotter->impl->CreateStream(keywords));
  return stream;
}

int32_t SherpaOnnxIsKeywordStreamReady(const SherpaOnnxKeywordSpotter *spotter,
                                       const SherpaOnnxOnlineStream *stream) {
  return spotter->impl->IsReady(stream->impl.get());
}

void SherpaOnnxDecodeKeywordStream(const SherpaOnnxKeywordSpotter *spotter,
                                   const SherpaOnnxOnlineStream *stream) {
  spotter->impl->DecodeStream(stream->impl.get());
}

void SherpaOnnxResetKeywordStream(const SherpaOnnxKeywordSpotter *spotter,
                                  const SherpaOnnxOnlineStream *stream) {
  spotter->impl->Reset(stream->impl.get());
}

void SherpaOnnxDecodeMultipleKeywordStreams(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream **streams, int32_t n) {
  std::vector<sherpa_onnx::OnlineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  spotter->impl->DecodeStreams(ss.data(), n);
}

const SherpaOnnxKeywordResult *SherpaOnnxGetKeywordResult(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream *stream) {
  const sherpa_onnx::KeywordResult &result =
      spotter->impl->GetResult(stream->impl.get());
  const auto &keyword = result.keyword;

  auto r = new SherpaOnnxKeywordResult;
  memset(r, 0, sizeof(SherpaOnnxKeywordResult));

  r->start_time = result.start_time;

  // copy keyword
  char *pKeyword = new char[keyword.size() + 1];
  std::copy(keyword.begin(), keyword.end(), pKeyword);
  pKeyword[keyword.size()] = 0;
  r->keyword = pKeyword;

  // copy json
  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  r->json = pJson;

  // copy tokens
  auto count = result.tokens.size();
  if (count > 0) {
    size_t total_length = 0;
    for (const auto &token : result.tokens) {
      // +1 for the null character at the end of each token
      total_length += token.size() + 1;
    }

    r->count = count;
    // Each word ends with nullptr
    char *pTokens = new char[total_length]{};
    char **tokens_temp = new char *[r->count];
    int32_t pos = 0;
    for (int32_t i = 0; i < r->count; ++i) {
      tokens_temp[i] = pTokens + pos;
      memcpy(pTokens + pos, result.tokens[i].c_str(), result.tokens[i].size());
      // +1 to move past the null character
      pos += result.tokens[i].size() + 1;
    }
    r->tokens = pTokens;
    r->tokens_arr = tokens_temp;

    if (!result.timestamps.empty()) {
      r->timestamps = new float[result.timestamps.size()];
      std::copy(result.timestamps.begin(), result.timestamps.end(),
                r->timestamps);
    } else {
      r->timestamps = nullptr;
    }

  } else {
    r->count = 0;
    r->timestamps = nullptr;
    r->tokens = nullptr;
    r->tokens_arr = nullptr;
  }

  return r;
}

void SherpaOnnxDestroyKeywordResult(const SherpaOnnxKeywordResult *r) {
  if (r) {
    delete[] r->keyword;
    delete[] r->json;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->timestamps;
    delete r;
  }
}

const char *SherpaOnnxGetKeywordResultAsJson(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream *stream) {
  const sherpa_onnx::KeywordResult &result =
      spotter->impl->GetResult(stream->impl.get());

  std::string json = result.AsJsonString();
  char *pJson = new char[json.size() + 1];
  std::copy(json.begin(), json.end(), pJson);
  pJson[json.size()] = 0;
  return pJson;
}

void SherpaOnnxFreeKeywordResultJson(const char *s) { delete[] s; }

// ============================================================
// For VAD
// ============================================================
//
struct SherpaOnnxCircularBuffer {
  std::unique_ptr<sherpa_onnx::CircularBuffer> impl;
};

const SherpaOnnxCircularBuffer *SherpaOnnxCreateCircularBuffer(
    int32_t capacity) {
  SherpaOnnxCircularBuffer *buffer = new SherpaOnnxCircularBuffer;
  buffer->impl = std::make_unique<sherpa_onnx::CircularBuffer>(capacity);
  return buffer;
}

void SherpaOnnxDestroyCircularBuffer(const SherpaOnnxCircularBuffer *buffer) {
  delete buffer;
}

void SherpaOnnxCircularBufferPush(const SherpaOnnxCircularBuffer *buffer,
                                  const float *p, int32_t n) {
  buffer->impl->Push(p, n);
}

const float *SherpaOnnxCircularBufferGet(const SherpaOnnxCircularBuffer *buffer,
                                         int32_t start_index, int32_t n) {
  std::vector<float> v = buffer->impl->Get(start_index, n);

  float *p = new float[n];
  std::copy(v.begin(), v.end(), p);
  return p;
}

void SherpaOnnxCircularBufferFree(const float *p) { delete[] p; }

void SherpaOnnxCircularBufferPop(const SherpaOnnxCircularBuffer *buffer,
                                 int32_t n) {
  buffer->impl->Pop(n);
}

int32_t SherpaOnnxCircularBufferSize(const SherpaOnnxCircularBuffer *buffer) {
  return buffer->impl->Size();
}

int32_t SherpaOnnxCircularBufferHead(const SherpaOnnxCircularBuffer *buffer) {
  return buffer->impl->Head();
}

void SherpaOnnxCircularBufferReset(const SherpaOnnxCircularBuffer *buffer) {
  buffer->impl->Reset();
}

struct SherpaOnnxVoiceActivityDetector {
  std::unique_ptr<sherpa_onnx::VoiceActivityDetector> impl;
};

sherpa_onnx::VadModelConfig GetVadModelConfig(
    const SherpaOnnxVadModelConfig *config) {
  sherpa_onnx::VadModelConfig vad_config;

  vad_config.silero_vad.model = SHERPA_ONNX_OR(config->silero_vad.model, "");
  vad_config.silero_vad.threshold =
      SHERPA_ONNX_OR(config->silero_vad.threshold, 0.5);

  vad_config.silero_vad.min_silence_duration =
      SHERPA_ONNX_OR(config->silero_vad.min_silence_duration, 0.5);

  vad_config.silero_vad.min_speech_duration =
      SHERPA_ONNX_OR(config->silero_vad.min_speech_duration, 0.25);

  vad_config.silero_vad.window_size =
      SHERPA_ONNX_OR(config->silero_vad.window_size, 512);

  vad_config.silero_vad.max_speech_duration =
      SHERPA_ONNX_OR(config->silero_vad.max_speech_duration, 20);

  vad_config.ten_vad.model = SHERPA_ONNX_OR(config->ten_vad.model, "");
  vad_config.ten_vad.threshold = SHERPA_ONNX_OR(config->ten_vad.threshold, 0.5);

  vad_config.ten_vad.min_silence_duration =
      SHERPA_ONNX_OR(config->ten_vad.min_silence_duration, 0.5);

  vad_config.ten_vad.min_speech_duration =
      SHERPA_ONNX_OR(config->ten_vad.min_speech_duration, 0.25);

  vad_config.ten_vad.window_size =
      SHERPA_ONNX_OR(config->ten_vad.window_size, 256);

  vad_config.ten_vad.max_speech_duration =
      SHERPA_ONNX_OR(config->ten_vad.max_speech_duration, 20);

  vad_config.sample_rate = SHERPA_ONNX_OR(config->sample_rate, 16000);
  vad_config.num_threads = SHERPA_ONNX_OR(config->num_threads, 1);
  vad_config.provider = SHERPA_ONNX_OR(config->provider, "cpu");
  if (vad_config.provider.empty()) {
    vad_config.provider = "cpu";
  }

  vad_config.debug = SHERPA_ONNX_OR(config->debug, false);

  if (vad_config.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", vad_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", vad_config.ToString().c_str());
#endif
  }

  return vad_config;
}

const SherpaOnnxVoiceActivityDetector *SherpaOnnxCreateVoiceActivityDetector(
    const SherpaOnnxVadModelConfig *config, float buffer_size_in_seconds) {
  auto vad_config = GetVadModelConfig(config);

  if (!vad_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxVoiceActivityDetector *p = new SherpaOnnxVoiceActivityDetector;
  p->impl = std::make_unique<sherpa_onnx::VoiceActivityDetector>(
      vad_config, buffer_size_in_seconds);

  return p;
}

void SherpaOnnxDestroyVoiceActivityDetector(
    const SherpaOnnxVoiceActivityDetector *p) {
  delete p;
}

void SherpaOnnxVoiceActivityDetectorAcceptWaveform(
    const SherpaOnnxVoiceActivityDetector *p, const float *samples, int32_t n) {
  p->impl->AcceptWaveform(samples, n);
}

int32_t SherpaOnnxVoiceActivityDetectorEmpty(
    const SherpaOnnxVoiceActivityDetector *p) {
  return p->impl->Empty();
}

int32_t SherpaOnnxVoiceActivityDetectorDetected(
    const SherpaOnnxVoiceActivityDetector *p) {
  return p->impl->IsSpeechDetected();
}

void SherpaOnnxVoiceActivityDetectorPop(
    const SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Pop();
}

void SherpaOnnxVoiceActivityDetectorClear(
    const SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Clear();
}

const SherpaOnnxSpeechSegment *SherpaOnnxVoiceActivityDetectorFront(
    const SherpaOnnxVoiceActivityDetector *p) {
  const sherpa_onnx::SpeechSegment &segment = p->impl->Front();

  SherpaOnnxSpeechSegment *ans = new SherpaOnnxSpeechSegment;
  ans->start = segment.start;
  ans->samples = new float[segment.samples.size()];
  std::copy(segment.samples.begin(), segment.samples.end(), ans->samples);
  ans->n = segment.samples.size();

  return ans;
}

void SherpaOnnxDestroySpeechSegment(const SherpaOnnxSpeechSegment *p) {
  if (p) {
    delete[] p->samples;
    delete p;
  }
}

void SherpaOnnxVoiceActivityDetectorReset(
    const SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Reset();
}

void SherpaOnnxVoiceActivityDetectorFlush(
    const SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Flush();
}

#if SHERPA_ONNX_ENABLE_TTS == 1
struct SherpaOnnxOfflineTts {
  std::unique_ptr<sherpa_onnx::OfflineTts> impl;
};

static sherpa_onnx::OfflineTtsConfig GetOfflineTtsConfig(
    const SherpaOnnxOfflineTtsConfig *config) {
  sherpa_onnx::OfflineTtsConfig tts_config;

  // vits
  tts_config.model.vits.model = SHERPA_ONNX_OR(config->model.vits.model, "");
  tts_config.model.vits.lexicon =
      SHERPA_ONNX_OR(config->model.vits.lexicon, "");
  tts_config.model.vits.tokens = SHERPA_ONNX_OR(config->model.vits.tokens, "");
  tts_config.model.vits.data_dir =
      SHERPA_ONNX_OR(config->model.vits.data_dir, "");
  tts_config.model.vits.noise_scale =
      SHERPA_ONNX_OR(config->model.vits.noise_scale, 0.667);
  tts_config.model.vits.noise_scale_w =
      SHERPA_ONNX_OR(config->model.vits.noise_scale_w, 0.8);
  tts_config.model.vits.length_scale =
      SHERPA_ONNX_OR(config->model.vits.length_scale, 1.0);
  tts_config.model.vits.dict_dir =
      SHERPA_ONNX_OR(config->model.vits.dict_dir, "");

  // matcha
  tts_config.model.matcha.acoustic_model =
      SHERPA_ONNX_OR(config->model.matcha.acoustic_model, "");
  tts_config.model.matcha.vocoder =
      SHERPA_ONNX_OR(config->model.matcha.vocoder, "");
  tts_config.model.matcha.lexicon =
      SHERPA_ONNX_OR(config->model.matcha.lexicon, "");
  tts_config.model.matcha.tokens =
      SHERPA_ONNX_OR(config->model.matcha.tokens, "");
  tts_config.model.matcha.data_dir =
      SHERPA_ONNX_OR(config->model.matcha.data_dir, "");
  tts_config.model.matcha.noise_scale =
      SHERPA_ONNX_OR(config->model.matcha.noise_scale, 0.667);
  tts_config.model.matcha.length_scale =
      SHERPA_ONNX_OR(config->model.matcha.length_scale, 1.0);
  tts_config.model.matcha.dict_dir =
      SHERPA_ONNX_OR(config->model.matcha.dict_dir, "");

  // kokoro
  tts_config.model.kokoro.model =
      SHERPA_ONNX_OR(config->model.kokoro.model, "");
  tts_config.model.kokoro.voices =
      SHERPA_ONNX_OR(config->model.kokoro.voices, "");
  tts_config.model.kokoro.tokens =
      SHERPA_ONNX_OR(config->model.kokoro.tokens, "");
  tts_config.model.kokoro.data_dir =
      SHERPA_ONNX_OR(config->model.kokoro.data_dir, "");
  tts_config.model.kokoro.length_scale =
      SHERPA_ONNX_OR(config->model.kokoro.length_scale, 1.0);
  tts_config.model.kokoro.dict_dir =
      SHERPA_ONNX_OR(config->model.kokoro.dict_dir, "");
  tts_config.model.kokoro.lexicon =
      SHERPA_ONNX_OR(config->model.kokoro.lexicon, "");
  tts_config.model.kokoro.lang = SHERPA_ONNX_OR(config->model.kokoro.lang, "");

  tts_config.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  tts_config.model.debug = config->model.debug;
  tts_config.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");
  if (tts_config.model.provider.empty()) {
    tts_config.model.provider = "cpu";
  }

  tts_config.rule_fsts = SHERPA_ONNX_OR(config->rule_fsts, "");
  tts_config.rule_fars = SHERPA_ONNX_OR(config->rule_fars, "");
  tts_config.max_num_sentences = SHERPA_ONNX_OR(config->max_num_sentences, 1);
  tts_config.silence_scale = SHERPA_ONNX_OR(config->silence_scale, 0.2);

  if (tts_config.model.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", tts_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", tts_config.ToString().c_str());
#endif
  }

  return tts_config;
}

const SherpaOnnxOfflineTts *SherpaOnnxCreateOfflineTts(
    const SherpaOnnxOfflineTtsConfig *config) {
  auto tts_config = GetOfflineTtsConfig(config);

  if (!tts_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxOfflineTts *tts = new SherpaOnnxOfflineTts;

  tts->impl = std::make_unique<sherpa_onnx::OfflineTts>(tts_config);

  return tts;
}

void SherpaOnnxDestroyOfflineTts(const SherpaOnnxOfflineTts *tts) {
  delete tts;
}

int32_t SherpaOnnxOfflineTtsSampleRate(const SherpaOnnxOfflineTts *tts) {
  return tts->impl->SampleRate();
}

int32_t SherpaOnnxOfflineTtsNumSpeakers(const SherpaOnnxOfflineTts *tts) {
  return tts->impl->NumSpeakers();
}

static const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateInternal(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    std::function<int32_t(const float *, int32_t, float)> callback) {
  sherpa_onnx::GeneratedAudio audio =
      tts->impl->Generate(text, sid, speed, callback);

  if (audio.samples.empty()) {
    return nullptr;
  }

  SherpaOnnxGeneratedAudio *ans = new SherpaOnnxGeneratedAudio;

  float *samples = new float[audio.samples.size()];
  std::copy(audio.samples.begin(), audio.samples.end(), samples);

  ans->samples = samples;
  ans->n = audio.samples.size();
  ans->sample_rate = audio.sample_rate;

  return ans;
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerate(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid,
    float speed) {
  return SherpaOnnxOfflineTtsGenerateInternal(tts, text, sid, speed, nullptr);
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateWithCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallback callback) {
  auto wrapper = [callback](const float *samples, int32_t n,
                            float /*progress*/) {
    return callback(samples, n);
  };

  return SherpaOnnxOfflineTtsGenerateInternal(tts, text, sid, speed, wrapper);
}

const SherpaOnnxGeneratedAudio *
SherpaOnnxOfflineTtsGenerateWithProgressCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioProgressCallback callback) {
  auto wrapper = [callback](const float *samples, int32_t n, float progress) {
    return callback(samples, n, progress);
  };
  return SherpaOnnxOfflineTtsGenerateInternal(tts, text, sid, speed, wrapper);
}

const SherpaOnnxGeneratedAudio *
SherpaOnnxOfflineTtsGenerateWithProgressCallbackWithArg(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioProgressCallbackWithArg callback, void *arg) {
  auto wrapper = [callback, arg](const float *samples, int32_t n,
                                 float progress) {
    return callback(samples, n, progress, arg);
  };
  return SherpaOnnxOfflineTtsGenerateInternal(tts, text, sid, speed, wrapper);
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateWithCallbackWithArg(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallbackWithArg callback, void *arg) {
  auto wrapper = [callback, arg](const float *samples, int32_t n,
                                 float /*progress*/) {
    return callback(samples, n, arg);
  };

  return SherpaOnnxOfflineTtsGenerateInternal(tts, text, sid, speed, wrapper);
}

void SherpaOnnxDestroyOfflineTtsGeneratedAudio(
    const SherpaOnnxGeneratedAudio *p) {
  if (p) {
    delete[] p->samples;
    delete p;
  }
}
#else
const SherpaOnnxOfflineTts *SherpaOnnxCreateOfflineTts(
    const SherpaOnnxOfflineTtsConfig *config) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

void SherpaOnnxDestroyOfflineTts(const SherpaOnnxOfflineTts *tts) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
}

int32_t SherpaOnnxOfflineTtsSampleRate(const SherpaOnnxOfflineTts *tts) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
  return 0;
}

int32_t SherpaOnnxOfflineTtsNumSpeakers(const SherpaOnnxOfflineTts *tts) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
  return 0;
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerate(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid,
    float speed) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateWithCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallback callback) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

const SherpaOnnxGeneratedAudio *
SherpaOnnxOfflineTtsGenerateWithProgressCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioProgressCallback callback) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

const SherpaOnnxGeneratedAudio *
SherpaOnnxOfflineTtsGenerateWithProgressCallbackWithArg(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioProgressCallbackWithArg callback, void *arg) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateWithCallbackWithArg(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallbackWithArg callback, void *arg) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

void SherpaOnnxDestroyOfflineTtsGeneratedAudio(
    const SherpaOnnxGeneratedAudio *p) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
}

#endif  // SHERPA_ONNX_ENABLE_TTS == 1

int32_t SherpaOnnxWriteWave(const float *samples, int32_t n,
                            int32_t sample_rate, const char *filename) {
  return sherpa_onnx::WriteWave(filename, sample_rate, samples, n);
}

int64_t SherpaOnnxWaveFileSize(int32_t n_samples) {
  return sherpa_onnx::WaveFileSize(n_samples);
}

void SherpaOnnxWriteWaveToBuffer(const float *samples, int32_t n,
                                 int32_t sample_rate, char *buffer) {
  sherpa_onnx::WriteWave(buffer, sample_rate, samples, n);
}

const SherpaOnnxWave *SherpaOnnxReadWave(const char *filename) {
  int32_t sample_rate = -1;
  bool is_ok = false;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(filename, &sample_rate, &is_ok);
  if (!is_ok) {
    return nullptr;
  }

  float *c_samples = new float[samples.size()];
  std::copy(samples.begin(), samples.end(), c_samples);

  SherpaOnnxWave *wave = new SherpaOnnxWave;
  wave->samples = c_samples;
  wave->sample_rate = sample_rate;
  wave->num_samples = samples.size();
  return wave;
}

const SherpaOnnxWave *SherpaOnnxReadWaveFromBinaryData(const char *data,
                                                       int32_t n) {
  int32_t sample_rate = -1;
  bool is_ok = false;

  std::istrstream is(data, n);

  std::vector<float> samples = sherpa_onnx::ReadWave(is, &sample_rate, &is_ok);
  if (!is_ok) {
    return nullptr;
  }

  float *c_samples = new float[samples.size()];
  std::copy(samples.begin(), samples.end(), c_samples);

  SherpaOnnxWave *wave = new SherpaOnnxWave;
  wave->samples = c_samples;
  wave->sample_rate = sample_rate;
  wave->num_samples = samples.size();
  return wave;
}

void SherpaOnnxFreeWave(const SherpaOnnxWave *wave) {
  if (wave) {
    delete[] wave->samples;
    delete wave;
  }
}

struct SherpaOnnxSpokenLanguageIdentification {
  std::unique_ptr<sherpa_onnx::SpokenLanguageIdentification> impl;
};

const SherpaOnnxSpokenLanguageIdentification *
SherpaOnnxCreateSpokenLanguageIdentification(
    const SherpaOnnxSpokenLanguageIdentificationConfig *config) {
  sherpa_onnx::SpokenLanguageIdentificationConfig slid_config;
  slid_config.whisper.encoder = SHERPA_ONNX_OR(config->whisper.encoder, "");
  slid_config.whisper.decoder = SHERPA_ONNX_OR(config->whisper.decoder, "");
  slid_config.whisper.tail_paddings =
      SHERPA_ONNX_OR(config->whisper.tail_paddings, -1);
  slid_config.num_threads = SHERPA_ONNX_OR(config->num_threads, 1);
  slid_config.debug = config->debug;
  slid_config.provider = SHERPA_ONNX_OR(config->provider, "cpu");
  if (slid_config.provider.empty()) {
    slid_config.provider = "cpu";
  }

  if (slid_config.debug) {
    SHERPA_ONNX_LOGE("%s\n", slid_config.ToString().c_str());
  }

  if (!slid_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxSpokenLanguageIdentification *slid =
      new SherpaOnnxSpokenLanguageIdentification;
  slid->impl =
      std::make_unique<sherpa_onnx::SpokenLanguageIdentification>(slid_config);

  return slid;
}

void SherpaOnnxDestroySpokenLanguageIdentification(
    const SherpaOnnxSpokenLanguageIdentification *slid) {
  delete slid;
}

SherpaOnnxOfflineStream *
SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(
    const SherpaOnnxSpokenLanguageIdentification *slid) {
  SherpaOnnxOfflineStream *stream =
      new SherpaOnnxOfflineStream(slid->impl->CreateStream());
  return stream;
}

const SherpaOnnxSpokenLanguageIdentificationResult *
SherpaOnnxSpokenLanguageIdentificationCompute(
    const SherpaOnnxSpokenLanguageIdentification *slid,
    const SherpaOnnxOfflineStream *s) {
  std::string lang = slid->impl->Compute(s->impl.get());
  char *c_lang = new char[lang.size() + 1];
  std::copy(lang.begin(), lang.end(), c_lang);
  c_lang[lang.size()] = '\0';
  SherpaOnnxSpokenLanguageIdentificationResult *r =
      new SherpaOnnxSpokenLanguageIdentificationResult;
  r->lang = c_lang;
  return r;
}

void SherpaOnnxDestroySpokenLanguageIdentificationResult(
    const SherpaOnnxSpokenLanguageIdentificationResult *r) {
  if (r) {
    delete[] r->lang;
    delete r;
  }
}

struct SherpaOnnxSpeakerEmbeddingExtractor {
  std::unique_ptr<sherpa_onnx::SpeakerEmbeddingExtractor> impl;
};

static sherpa_onnx::SpeakerEmbeddingExtractorConfig
GetSpeakerEmbeddingExtractorConfig(
    const SherpaOnnxSpeakerEmbeddingExtractorConfig *config) {
  sherpa_onnx::SpeakerEmbeddingExtractorConfig c;
  c.model = SHERPA_ONNX_OR(config->model, "");

  c.num_threads = SHERPA_ONNX_OR(config->num_threads, 1);
  c.debug = SHERPA_ONNX_OR(config->debug, 0);
  c.provider = SHERPA_ONNX_OR(config->provider, "cpu");
  if (c.provider.empty()) {
    c.provider = "cpu";
  }

  if (config->debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", c.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", c.ToString().c_str());
#endif
  }

  return c;
}

const SherpaOnnxSpeakerEmbeddingExtractor *
SherpaOnnxCreateSpeakerEmbeddingExtractor(
    const SherpaOnnxSpeakerEmbeddingExtractorConfig *config) {
  auto c = GetSpeakerEmbeddingExtractorConfig(config);

  if (!c.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config!");
    return nullptr;
  }

  auto p = new SherpaOnnxSpeakerEmbeddingExtractor;

  p->impl = std::make_unique<sherpa_onnx::SpeakerEmbeddingExtractor>(c);

  return p;
}

void SherpaOnnxDestroySpeakerEmbeddingExtractor(
    const SherpaOnnxSpeakerEmbeddingExtractor *p) {
  delete p;
}

int32_t SherpaOnnxSpeakerEmbeddingExtractorDim(
    const SherpaOnnxSpeakerEmbeddingExtractor *p) {
  return p->impl->Dim();
}

const SherpaOnnxOnlineStream *SherpaOnnxSpeakerEmbeddingExtractorCreateStream(
    const SherpaOnnxSpeakerEmbeddingExtractor *p) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(p->impl->CreateStream());
  return stream;
}

int32_t SherpaOnnxSpeakerEmbeddingExtractorIsReady(
    const SherpaOnnxSpeakerEmbeddingExtractor *p,
    const SherpaOnnxOnlineStream *s) {
  return p->impl->IsReady(s->impl.get());
}

const float *SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(
    const SherpaOnnxSpeakerEmbeddingExtractor *p,
    const SherpaOnnxOnlineStream *s) {
  std::vector<float> v = p->impl->Compute(s->impl.get());
  float *ans = new float[v.size()];
  std::copy(v.begin(), v.end(), ans);
  return ans;
}

void SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(const float *v) {
  delete[] v;
}

struct SherpaOnnxSpeakerEmbeddingManager {
  std::unique_ptr<sherpa_onnx::SpeakerEmbeddingManager> impl;
};

const SherpaOnnxSpeakerEmbeddingManager *
SherpaOnnxCreateSpeakerEmbeddingManager(int32_t dim) {
  auto p = new SherpaOnnxSpeakerEmbeddingManager;
  p->impl = std::make_unique<sherpa_onnx::SpeakerEmbeddingManager>(dim);
  return p;
}

void SherpaOnnxDestroySpeakerEmbeddingManager(
    const SherpaOnnxSpeakerEmbeddingManager *p) {
  delete p;
}

int32_t SherpaOnnxSpeakerEmbeddingManagerAdd(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name,
    const float *v) {
  return p->impl->Add(name, v);
}

int32_t SherpaOnnxSpeakerEmbeddingManagerAddList(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name,
    const float **v) {
  int32_t n = 0;
  auto q = v;
  while (q && q[0]) {
    ++n;
    ++q;
  }

  if (n == 0) {
    SHERPA_ONNX_LOGE("Empty embedding!");
    return 0;
  }

  std::vector<std::vector<float>> vec(n);
  int32_t dim = p->impl->Dim();

  for (int32_t i = 0; i != n; ++i) {
    vec[i] = std::vector<float>(v[i], v[i] + dim);
  }

  return p->impl->Add(name, vec);
}

int32_t SherpaOnnxSpeakerEmbeddingManagerAddListFlattened(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name,
    const float *v, int32_t n) {
  std::vector<std::vector<float>> vec(n);

  int32_t dim = p->impl->Dim();

  for (int32_t i = 0; i != n; ++i, v += dim) {
    vec[i] = std::vector<float>(v, v + dim);
  }

  return p->impl->Add(name, vec);
}

int32_t SherpaOnnxSpeakerEmbeddingManagerRemove(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name) {
  return p->impl->Remove(name);
}

const char *SherpaOnnxSpeakerEmbeddingManagerSearch(
    const SherpaOnnxSpeakerEmbeddingManager *p, const float *v,
    float threshold) {
  auto r = p->impl->Search(v, threshold);
  if (r.empty()) {
    return nullptr;
  }

  char *name = new char[r.size() + 1];
  std::copy(r.begin(), r.end(), name);
  name[r.size()] = '\0';

  return name;
}

void SherpaOnnxSpeakerEmbeddingManagerFreeSearch(const char *name) {
  delete[] name;
}

const SherpaOnnxSpeakerEmbeddingManagerBestMatchesResult *
SherpaOnnxSpeakerEmbeddingManagerGetBestMatches(
    const SherpaOnnxSpeakerEmbeddingManager *p, const float *v, float threshold,
    int32_t n) {
  auto matches = p->impl->GetBestMatches(v, threshold, n);

  if (matches.empty()) {
    return nullptr;
  }

  auto resultMatches =
      new SherpaOnnxSpeakerEmbeddingManagerSpeakerMatch[matches.size()];
  for (int i = 0; i < matches.size(); ++i) {
    resultMatches[i].score = matches[i].score;

    char *name = new char[matches[i].name.size() + 1];
    std::copy(matches[i].name.begin(), matches[i].name.end(), name);
    name[matches[i].name.size()] = '\0';

    resultMatches[i].name = name;
  }

  auto *result = new SherpaOnnxSpeakerEmbeddingManagerBestMatchesResult();
  result->count = matches.size();
  result->matches = resultMatches;

  return result;
}

void SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches(
    const SherpaOnnxSpeakerEmbeddingManagerBestMatchesResult *r) {
  if (r == nullptr) {
    return;
  }

  for (int32_t i = 0; i < r->count; ++i) {
    delete[] r->matches[i].name;
  }
  delete[] r->matches;
  delete r;
}

int32_t SherpaOnnxSpeakerEmbeddingManagerVerify(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name,
    const float *v, float threshold) {
  return p->impl->Verify(name, v, threshold);
}

int32_t SherpaOnnxSpeakerEmbeddingManagerContains(
    const SherpaOnnxSpeakerEmbeddingManager *p, const char *name) {
  return p->impl->Contains(name);
}

int32_t SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(
    const SherpaOnnxSpeakerEmbeddingManager *p) {
  return p->impl->NumSpeakers();
}

const char *const *SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(
    const SherpaOnnxSpeakerEmbeddingManager *manager) {
  std::vector<std::string> all_speakers = manager->impl->GetAllSpeakers();
  int32_t num_speakers = all_speakers.size();
  char **p = new char *[num_speakers + 1];
  p[num_speakers] = nullptr;

  int32_t i = 0;
  for (const auto &name : all_speakers) {
    p[i] = new char[name.size() + 1];
    std::copy(name.begin(), name.end(), p[i]);
    p[i][name.size()] = '\0';

    i += 1;
  }
  return p;
}

void SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(
    const char *const *names) {
  auto p = names;

  while (p && p[0]) {
    delete[] p[0];
    ++p;
  }

  delete[] names;
}

struct SherpaOnnxAudioTagging {
  std::unique_ptr<sherpa_onnx::AudioTagging> impl;
};

const SherpaOnnxAudioTagging *SherpaOnnxCreateAudioTagging(
    const SherpaOnnxAudioTaggingConfig *config) {
  sherpa_onnx::AudioTaggingConfig ac;
  ac.model.zipformer.model = SHERPA_ONNX_OR(config->model.zipformer.model, "");
  ac.model.ced = SHERPA_ONNX_OR(config->model.ced, "");
  ac.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  ac.model.debug = config->model.debug;
  ac.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");
  if (ac.model.provider.empty()) {
    ac.model.provider = "cpu";
  }

  ac.labels = SHERPA_ONNX_OR(config->labels, "");
  ac.top_k = SHERPA_ONNX_OR(config->top_k, 5);

  if (ac.model.debug) {
    SHERPA_ONNX_LOGE("%s\n", ac.ToString().c_str());
  }

  if (!ac.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxAudioTagging *tagger = new SherpaOnnxAudioTagging;
  tagger->impl = std::make_unique<sherpa_onnx::AudioTagging>(ac);

  return tagger;
}

void SherpaOnnxDestroyAudioTagging(const SherpaOnnxAudioTagging *tagger) {
  delete tagger;
}

const SherpaOnnxOfflineStream *SherpaOnnxAudioTaggingCreateOfflineStream(
    const SherpaOnnxAudioTagging *tagger) {
  const SherpaOnnxOfflineStream *stream =
      new SherpaOnnxOfflineStream(tagger->impl->CreateStream());
  return stream;
}

const SherpaOnnxAudioEvent *const *SherpaOnnxAudioTaggingCompute(
    const SherpaOnnxAudioTagging *tagger, const SherpaOnnxOfflineStream *s,
    int32_t top_k) {
  std::vector<sherpa_onnx::AudioEvent> events =
      tagger->impl->Compute(s->impl.get(), top_k);

  int32_t n = static_cast<int32_t>(events.size());
  SherpaOnnxAudioEvent **ans = new SherpaOnnxAudioEvent *[n + 1];
  ans[n] = nullptr;

  int32_t i = 0;
  for (const auto &e : events) {
    SherpaOnnxAudioEvent *p = new SherpaOnnxAudioEvent;

    char *name = new char[e.name.size() + 1];
    std::copy(e.name.begin(), e.name.end(), name);
    name[e.name.size()] = 0;

    p->name = name;

    p->index = e.index;
    p->prob = e.prob;

    ans[i] = p;
    i += 1;
  }

  return ans;
}

void SherpaOnnxAudioTaggingFreeResults(
    const SherpaOnnxAudioEvent *const *events) {
  auto p = events;

  while (p && *p) {
    auto e = *p;

    delete[] e->name;
    delete e;

    ++p;
  }

  delete[] events;
}

struct SherpaOnnxOfflinePunctuation {
  std::unique_ptr<sherpa_onnx::OfflinePunctuation> impl;
};

const SherpaOnnxOfflinePunctuation *SherpaOnnxCreateOfflinePunctuation(
    const SherpaOnnxOfflinePunctuationConfig *config) {
  sherpa_onnx::OfflinePunctuationConfig c;
  c.model.ct_transformer = SHERPA_ONNX_OR(config->model.ct_transformer, "");
  c.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  c.model.debug = config->model.debug;
  c.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");
  if (c.model.provider.empty()) {
    c.model.provider = "cpu";
  }

  if (c.model.debug) {
    SHERPA_ONNX_LOGE("%s\n", c.ToString().c_str());
  }

  if (!c.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxOfflinePunctuation *punct = new SherpaOnnxOfflinePunctuation;
  punct->impl = std::make_unique<sherpa_onnx::OfflinePunctuation>(c);

  return punct;
}

void SherpaOnnxDestroyOfflinePunctuation(
    const SherpaOnnxOfflinePunctuation *punct) {
  delete punct;
}

const char *SherpaOfflinePunctuationAddPunct(
    const SherpaOnnxOfflinePunctuation *punct, const char *text) {
  std::string text_with_punct = punct->impl->AddPunctuation(text);

  char *ans = new char[text_with_punct.size() + 1];
  std::copy(text_with_punct.begin(), text_with_punct.end(), ans);
  ans[text_with_punct.size()] = 0;

  return ans;
}

void SherpaOfflinePunctuationFreeText(const char *text) { delete[] text; }

struct SherpaOnnxOnlinePunctuation {
  std::unique_ptr<sherpa_onnx::OnlinePunctuation> impl;
};

const SherpaOnnxOnlinePunctuation *SherpaOnnxCreateOnlinePunctuation(
    const SherpaOnnxOnlinePunctuationConfig *config) {
  auto p = new SherpaOnnxOnlinePunctuation;
  try {
    sherpa_onnx::OnlinePunctuationConfig punctuation_config;
    punctuation_config.model.cnn_bilstm =
        SHERPA_ONNX_OR(config->model.cnn_bilstm, "");
    punctuation_config.model.bpe_vocab =
        SHERPA_ONNX_OR(config->model.bpe_vocab, "");
    punctuation_config.model.num_threads =
        SHERPA_ONNX_OR(config->model.num_threads, 1);
    punctuation_config.model.debug = config->model.debug;
    punctuation_config.model.provider =
        SHERPA_ONNX_OR(config->model.provider, "cpu");

    p->impl =
        std::make_unique<sherpa_onnx::OnlinePunctuation>(punctuation_config);
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("Failed to create online punctuation: %s", e.what());
    delete p;
    return nullptr;
  }
  return p;
}

void SherpaOnnxDestroyOnlinePunctuation(const SherpaOnnxOnlinePunctuation *p) {
  delete p;
}

const char *SherpaOnnxOnlinePunctuationAddPunct(
    const SherpaOnnxOnlinePunctuation *punctuation, const char *text) {
  if (!punctuation || !text) return nullptr;

  try {
    std::string s = punctuation->impl->AddPunctuationWithCase(text);
    char *p = new char[s.size() + 1];
    std::copy(s.begin(), s.end(), p);
    p[s.size()] = '\0';
    return p;
  } catch (const std::exception &e) {
    SHERPA_ONNX_LOGE("Failed to add punctuation: %s", e.what());
    return nullptr;
  }
}

void SherpaOnnxOnlinePunctuationFreeText(const char *text) { delete[] text; }

struct SherpaOnnxLinearResampler {
  std::unique_ptr<sherpa_onnx::LinearResample> impl;
};

const SherpaOnnxLinearResampler *SherpaOnnxCreateLinearResampler(
    int32_t samp_rate_in_hz, int32_t samp_rate_out_hz, float filter_cutoff_hz,
    int32_t num_zeros) {
  SherpaOnnxLinearResampler *p = new SherpaOnnxLinearResampler;
  p->impl = std::make_unique<sherpa_onnx::LinearResample>(
      samp_rate_in_hz, samp_rate_out_hz, filter_cutoff_hz, num_zeros);

  return p;
}

void SherpaOnnxDestroyLinearResampler(const SherpaOnnxLinearResampler *p) {
  delete p;
}

const SherpaOnnxResampleOut *SherpaOnnxLinearResamplerResample(
    const SherpaOnnxLinearResampler *p, const float *input, int32_t input_dim,
    int32_t flush) {
  std::vector<float> o;
  p->impl->Resample(input, input_dim, flush, &o);

  float *s = new float[o.size()];
  std::copy(o.begin(), o.end(), s);

  SherpaOnnxResampleOut *ans = new SherpaOnnxResampleOut;
  ans->samples = s;
  ans->n = static_cast<int32_t>(o.size());

  return ans;
}

void SherpaOnnxLinearResamplerResampleFree(const SherpaOnnxResampleOut *p) {
  delete[] p->samples;
  delete p;
}

int32_t SherpaOnnxLinearResamplerResampleGetInputSampleRate(
    const SherpaOnnxLinearResampler *p) {
  return p->impl->GetInputSamplingRate();
}

int32_t SherpaOnnxLinearResamplerResampleGetOutputSampleRate(
    const SherpaOnnxLinearResampler *p) {
  return p->impl->GetOutputSamplingRate();
}

void SherpaOnnxLinearResamplerReset(const SherpaOnnxLinearResampler *p) {
  p->impl->Reset();
}

int32_t SherpaOnnxFileExists(const char *filename) {
  return sherpa_onnx::FileExists(filename);
}

struct SherpaOnnxOfflineSpeechDenoiser {
  std::unique_ptr<sherpa_onnx::OfflineSpeechDenoiser> impl;
};

static sherpa_onnx::OfflineSpeechDenoiserConfig GetOfflineSpeechDenoiserConfig(
    const SherpaOnnxOfflineSpeechDenoiserConfig *config) {
  sherpa_onnx::OfflineSpeechDenoiserConfig c;
  c.model.gtcrn.model = SHERPA_ONNX_OR(config->model.gtcrn.model, "");
  c.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  c.model.debug = config->model.debug;
  c.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");

  if (c.model.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", c.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", c.ToString().c_str());
#endif
  }

  return c;
}

const SherpaOnnxOfflineSpeechDenoiser *SherpaOnnxCreateOfflineSpeechDenoiser(
    const SherpaOnnxOfflineSpeechDenoiserConfig *config) {
  auto sd_config = GetOfflineSpeechDenoiserConfig(config);

  if (!sd_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxOfflineSpeechDenoiser *sd = new SherpaOnnxOfflineSpeechDenoiser;

  sd->impl = std::make_unique<sherpa_onnx::OfflineSpeechDenoiser>(sd_config);

  return sd;
}

void SherpaOnnxDestroyOfflineSpeechDenoiser(
    const SherpaOnnxOfflineSpeechDenoiser *sd) {
  delete sd;
}

int32_t SherpaOnnxOfflineSpeechDenoiserGetSampleRate(
    const SherpaOnnxOfflineSpeechDenoiser *sd) {
  return sd->impl->GetSampleRate();
}

const SherpaOnnxDenoisedAudio *SherpaOnnxOfflineSpeechDenoiserRun(
    const SherpaOnnxOfflineSpeechDenoiser *sd, const float *samples, int32_t n,
    int32_t sample_rate) {
  auto audio = sd->impl->Run(samples, n, sample_rate);

  auto ans = new SherpaOnnxDenoisedAudio;

  float *denoised_samples = new float[audio.samples.size()];
  std::copy(audio.samples.begin(), audio.samples.end(), denoised_samples);

  ans->samples = denoised_samples;
  ans->n = audio.samples.size();
  ans->sample_rate = audio.sample_rate;

  return ans;
}

void SherpaOnnxDestroyDenoisedAudio(const SherpaOnnxDenoisedAudio *p) {
  delete[] p->samples;
  delete p;
}

#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1

struct SherpaOnnxOfflineSpeakerDiarization {
  std::unique_ptr<sherpa_onnx::OfflineSpeakerDiarization> impl;
};

struct SherpaOnnxOfflineSpeakerDiarizationResult {
  sherpa_onnx::OfflineSpeakerDiarizationResult impl;
};

static sherpa_onnx::OfflineSpeakerDiarizationConfig
GetOfflineSpeakerDiarizationConfig(
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config) {
  sherpa_onnx::OfflineSpeakerDiarizationConfig sd_config;

  sd_config.segmentation.pyannote.model =
      SHERPA_ONNX_OR(config->segmentation.pyannote.model, "");
  sd_config.segmentation.num_threads =
      SHERPA_ONNX_OR(config->segmentation.num_threads, 1);
  sd_config.segmentation.debug = config->segmentation.debug;
  sd_config.segmentation.provider =
      SHERPA_ONNX_OR(config->segmentation.provider, "cpu");
  if (sd_config.segmentation.provider.empty()) {
    sd_config.segmentation.provider = "cpu";
  }

  sd_config.embedding.model = SHERPA_ONNX_OR(config->embedding.model, "");
  sd_config.embedding.num_threads =
      SHERPA_ONNX_OR(config->embedding.num_threads, 1);
  sd_config.embedding.debug = config->embedding.debug;
  sd_config.embedding.provider =
      SHERPA_ONNX_OR(config->embedding.provider, "cpu");
  if (sd_config.embedding.provider.empty()) {
    sd_config.embedding.provider = "cpu";
  }

  sd_config.clustering.num_clusters =
      SHERPA_ONNX_OR(config->clustering.num_clusters, -1);

  sd_config.clustering.threshold =
      SHERPA_ONNX_OR(config->clustering.threshold, 0.5);

  sd_config.min_duration_on = SHERPA_ONNX_OR(config->min_duration_on, 0.3);

  sd_config.min_duration_off = SHERPA_ONNX_OR(config->min_duration_off, 0.5);

  if (sd_config.segmentation.debug || sd_config.embedding.debug) {
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", sd_config.ToString().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", sd_config.ToString().c_str());
#endif
  }

  return sd_config;
}

const SherpaOnnxOfflineSpeakerDiarization *
SherpaOnnxCreateOfflineSpeakerDiarization(
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config) {
  auto sd_config = GetOfflineSpeakerDiarizationConfig(config);

  if (!sd_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxOfflineSpeakerDiarization *sd =
      new SherpaOnnxOfflineSpeakerDiarization;

  sd->impl =
      std::make_unique<sherpa_onnx::OfflineSpeakerDiarization>(sd_config);

  return sd;
}

void SherpaOnnxDestroyOfflineSpeakerDiarization(
    const SherpaOnnxOfflineSpeakerDiarization *sd) {
  delete sd;
}

int32_t SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(
    const SherpaOnnxOfflineSpeakerDiarization *sd) {
  return sd->impl->SampleRate();
}

void SherpaOnnxOfflineSpeakerDiarizationSetConfig(
    const SherpaOnnxOfflineSpeakerDiarization *sd,
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config) {
  sherpa_onnx::OfflineSpeakerDiarizationConfig sd_config;

  sd_config.clustering.num_clusters =
      SHERPA_ONNX_OR(config->clustering.num_clusters, -1);

  sd_config.clustering.threshold =
      SHERPA_ONNX_OR(config->clustering.threshold, 0.5);

  sd->impl->SetConfig(sd_config);
}

int32_t SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r) {
  return r->impl.NumSpeakers();
}

int32_t SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r) {
  return r->impl.NumSegments();
}

const SherpaOnnxOfflineSpeakerDiarizationSegment *
SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r) {
  if (r->impl.NumSegments() == 0) {
    return nullptr;
  }

  auto segments = r->impl.SortByStartTime();

  int32_t n = segments.size();
  SherpaOnnxOfflineSpeakerDiarizationSegment *ans =
      new SherpaOnnxOfflineSpeakerDiarizationSegment[n];

  for (int32_t i = 0; i != n; ++i) {
    const auto &s = segments[i];

    ans[i].start = s.Start();
    ans[i].end = s.End();
    ans[i].speaker = s.Speaker();
  }

  return ans;
}

void SherpaOnnxOfflineSpeakerDiarizationDestroySegment(
    const SherpaOnnxOfflineSpeakerDiarizationSegment *s) {
  delete[] s;
}

const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcess(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n) {
  auto ans = new SherpaOnnxOfflineSpeakerDiarizationResult;
  ans->impl = sd->impl->Process(samples, n);

  return ans;
}

void SherpaOnnxOfflineSpeakerDiarizationDestroyResult(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r) {
  delete r;
}

const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n, SherpaOnnxOfflineSpeakerDiarizationProgressCallback callback,
    void *arg) {
  auto ans = new SherpaOnnxOfflineSpeakerDiarizationResult;
  ans->impl = sd->impl->Process(samples, n, callback, arg);

  return ans;
}

const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n,
    SherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg callback) {
  auto wrapper = [callback](int32_t num_processed_chunks,
                            int32_t num_total_chunks, void *) {
    return callback(num_processed_chunks, num_total_chunks);
  };

  auto ans = new SherpaOnnxOfflineSpeakerDiarizationResult;
  ans->impl = sd->impl->Process(samples, n, wrapper);

  return ans;
}
#else

const SherpaOnnxOfflineSpeakerDiarization *
SherpaOnnxCreateOfflineSpeakerDiarization(
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

void SherpaOnnxDestroyOfflineSpeakerDiarization(
    const SherpaOnnxOfflineSpeakerDiarization *sd) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
}

int32_t SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(
    const SherpaOnnxOfflineSpeakerDiarization *sd) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
  return 0;
}

void SherpaOnnxOfflineSpeakerDiarizationSetConfig(
    const SherpaOnnxOfflineSpeakerDiarization *sd,
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
}

int32_t SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
  return 0;
}

int32_t SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
  return 0;
}

const SherpaOnnxOfflineSpeakerDiarizationSegment *
SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

void SherpaOnnxOfflineSpeakerDiarizationDestroySegment(
    const SherpaOnnxOfflineSpeakerDiarizationSegment *s) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
}

const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcess(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n, SherpaOnnxOfflineSpeakerDiarizationProgressCallback callback,
    void *arg) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg(
    const SherpaOnnxOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n,
    SherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg callback) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

void SherpaOnnxOfflineSpeakerDiarizationDestroyResult(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
}

#endif

#ifdef __OHOS__

const SherpaOnnxOfflineSpeechDenoiser *
SherpaOnnxCreateOfflineSpeechDenoiserOHOS(
    const SherpaOnnxOfflineSpeechDenoiserConfig *config,
    NativeResourceManager *mgr) {
  auto sd_config = GetOfflineSpeechDenoiserConfig(config);

  SherpaOnnxOfflineSpeechDenoiser *sd = new SherpaOnnxOfflineSpeechDenoiser;

  sd->impl = std::make_unique<sherpa_onnx::OfflineSpeechDenoiser>(sd_config);

  return sd;
}

const SherpaOnnxOnlineRecognizer *SherpaOnnxCreateOnlineRecognizerOHOS(
    const SherpaOnnxOnlineRecognizerConfig *config,
    NativeResourceManager *mgr) {
  sherpa_onnx::OnlineRecognizerConfig recognizer_config =
      GetOnlineRecognizerConfig(config);

  SherpaOnnxOnlineRecognizer *recognizer = new SherpaOnnxOnlineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_onnx::OnlineRecognizer>(mgr, recognizer_config);

  return recognizer;
}

const SherpaOnnxOfflineRecognizer *SherpaOnnxCreateOfflineRecognizerOHOS(
    const SherpaOnnxOfflineRecognizerConfig *config,
    NativeResourceManager *mgr) {
  if (mgr == nullptr) {
    return SherpaOnnxCreateOfflineRecognizer(config);
  }

  sherpa_onnx::OfflineRecognizerConfig recognizer_config =
      GetOfflineRecognizerConfig(config);

  SherpaOnnxOfflineRecognizer *recognizer = new SherpaOnnxOfflineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_onnx::OfflineRecognizer>(mgr, recognizer_config);

  return recognizer;
}

const SherpaOnnxVoiceActivityDetector *
SherpaOnnxCreateVoiceActivityDetectorOHOS(
    const SherpaOnnxVadModelConfig *config, float buffer_size_in_seconds,
    NativeResourceManager *mgr) {
  if (mgr == nullptr) {
    return SherpaOnnxCreateVoiceActivityDetector(config,
                                                 buffer_size_in_seconds);
  }

  auto vad_config = GetVadModelConfig(config);

  SherpaOnnxVoiceActivityDetector *p = new SherpaOnnxVoiceActivityDetector;
  p->impl = std::make_unique<sherpa_onnx::VoiceActivityDetector>(
      mgr, vad_config, buffer_size_in_seconds);

  return p;
}

const SherpaOnnxSpeakerEmbeddingExtractor *
SherpaOnnxCreateSpeakerEmbeddingExtractorOHOS(
    const SherpaOnnxSpeakerEmbeddingExtractorConfig *config,
    NativeResourceManager *mgr) {
  if (!mgr) {
    return SherpaOnnxCreateSpeakerEmbeddingExtractor(config);
  }

  auto c = GetSpeakerEmbeddingExtractorConfig(config);

  auto p = new SherpaOnnxSpeakerEmbeddingExtractor;

  p->impl = std::make_unique<sherpa_onnx::SpeakerEmbeddingExtractor>(mgr, c);

  return p;
}

const SherpaOnnxKeywordSpotter *SherpaOnnxCreateKeywordSpotterOHOS(
    const SherpaOnnxKeywordSpotterConfig *config, NativeResourceManager *mgr) {
  if (!mgr) {
    return SherpaOnnxCreateKeywordSpotter(config);
  }

  auto spotter_config = GetKeywordSpotterConfig(config);

  SherpaOnnxKeywordSpotter *spotter = new SherpaOnnxKeywordSpotter;

  spotter->impl =
      std::make_unique<sherpa_onnx::KeywordSpotter>(mgr, spotter_config);

  return spotter;
}

#if SHERPA_ONNX_ENABLE_TTS == 1
const SherpaOnnxOfflineTts *SherpaOnnxCreateOfflineTtsOHOS(
    const SherpaOnnxOfflineTtsConfig *config, NativeResourceManager *mgr) {
  if (!mgr) {
    return SherpaOnnxCreateOfflineTts(config);
  }

  auto tts_config = GetOfflineTtsConfig(config);

  SherpaOnnxOfflineTts *tts = new SherpaOnnxOfflineTts;

  tts->impl = std::make_unique<sherpa_onnx::OfflineTts>(mgr, tts_config);

  return tts;
}
#else
const SherpaOnnxOfflineTts *SherpaOnnxCreateOfflineTtsOHOS(
    const SherpaOnnxOfflineTtsConfig *config, NativeResourceManager *mgr) {
  SHERPA_ONNX_LOGE("TTS is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}
#endif  // #if SHERPA_ONNX_ENABLE_TTS == 1
        //
#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1
const SherpaOnnxOfflineSpeakerDiarization *
SherpaOnnxCreateOfflineSpeakerDiarizationOHOS(
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config,
    NativeResourceManager *mgr) {
  if (!mgr) {
    return SherpaOnnxCreateOfflineSpeakerDiarization(config);
  }

  auto sd_config = GetOfflineSpeakerDiarizationConfig(config);

  SherpaOnnxOfflineSpeakerDiarization *sd =
      new SherpaOnnxOfflineSpeakerDiarization;

  sd->impl =
      std::make_unique<sherpa_onnx::OfflineSpeakerDiarization>(mgr, sd_config);

  return sd;
}
#else

const SherpaOnnxOfflineSpeakerDiarization *
SherpaOnnxCreateOfflineSpeakerDiarizationOHOS(
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config,
    NativeResourceManager *mgr) {
  SHERPA_ONNX_LOGE(
      "Speaker diarization is not enabled. Please rebuild sherpa-onnx");
  return nullptr;
}

#endif  // #if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1

#endif  // #ifdef __OHOS__
