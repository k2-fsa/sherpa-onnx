// sherpa-onnx/c-api/c-api.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/c-api/ezy-api.h"

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
#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

#if SHERPA_ONNX_ENABLE_TTS == 1
#include "sherpa-onnx/csrc/offline-tts.h"
#endif

#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1
#include "sherpa-onnx/csrc/offline-speaker-diarization.h"
#endif

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

void ResultBasic(
    int32_t *tokens, size_t *count,
    const SherpaOnnxOnlineRecognizer* recognizer,
    const SherpaOnnxOnlineStream* stream)
{
  sherpa_onnx::OnlineTransducerDecoderResult decoder_result =
      stream->impl->GetResult();
      //  decoder_->StripLeadingBlanks(&decoder_result);
//  return decoder_result.tokens;

  std::copy(decoder_result.tokens.begin(), decoder_result.tokens.end(),
            tokens);
  *count = decoder_result.tokens.size();
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
