// sherpa-onnx/c-api/c-api.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/c-api/c-api.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/circular-buffer.h"
#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/keyword-spotter.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/spoken-language-identification.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

#if SHERPA_ONNX_ENABLE_TTS == 1
#include "sherpa-onnx/csrc/offline-tts.h"
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

SherpaOnnxOnlineRecognizer *CreateOnlineRecognizer(
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
  recognizer_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  recognizer_config.model_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");
  recognizer_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");
  recognizer_config.model_config.debug =
      SHERPA_ONNX_OR(config->model_config.debug, 0);

  recognizer_config.decoding_method =
      SHERPA_ONNX_OR(config->decoding_method, "greedy_search");
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

  if (config->model_config.debug) {
    SHERPA_ONNX_LOGE("%s\n", recognizer_config.ToString().c_str());
  }

  if (!recognizer_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config!");
    return nullptr;
  }

  SherpaOnnxOnlineRecognizer *recognizer = new SherpaOnnxOnlineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_onnx::OnlineRecognizer>(recognizer_config);

  return recognizer;
}

void DestroyOnlineRecognizer(SherpaOnnxOnlineRecognizer *recognizer) {
  delete recognizer;
}

SherpaOnnxOnlineStream *CreateOnlineStream(
    const SherpaOnnxOnlineRecognizer *recognizer) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(recognizer->impl->CreateStream());
  return stream;
}

SherpaOnnxOnlineStream *CreateOnlineStreamWithHotwords(
    const SherpaOnnxOnlineRecognizer *recognizer, const char *hotwords) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(recognizer->impl->CreateStream(hotwords));
  return stream;
}

void DestroyOnlineStream(SherpaOnnxOnlineStream *stream) { delete stream; }

void AcceptWaveform(SherpaOnnxOnlineStream *stream, int32_t sample_rate,
                    const float *samples, int32_t n) {
  stream->impl->AcceptWaveform(sample_rate, samples, n);
}

int32_t IsOnlineStreamReady(SherpaOnnxOnlineRecognizer *recognizer,
                            SherpaOnnxOnlineStream *stream) {
  return recognizer->impl->IsReady(stream->impl.get());
}

void DecodeOnlineStream(SherpaOnnxOnlineRecognizer *recognizer,
                        SherpaOnnxOnlineStream *stream) {
  recognizer->impl->DecodeStream(stream->impl.get());
}

void DecodeMultipleOnlineStreams(SherpaOnnxOnlineRecognizer *recognizer,
                                 SherpaOnnxOnlineStream **streams, int32_t n) {
  std::vector<sherpa_onnx::OnlineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  recognizer->impl->DecodeStreams(ss.data(), n);
}

const SherpaOnnxOnlineRecognizerResult *GetOnlineStreamResult(
    SherpaOnnxOnlineRecognizer *recognizer, SherpaOnnxOnlineStream *stream) {
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
  const auto &json = result.AsJsonString();
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

    if (!result.timestamps.empty()) {
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

void DestroyOnlineRecognizerResult(const SherpaOnnxOnlineRecognizerResult *r) {
  if (r) {
    delete[] r->text;
    delete[] r->json;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->timestamps;
    delete r;
  }
}

void Reset(SherpaOnnxOnlineRecognizer *recognizer,
           SherpaOnnxOnlineStream *stream) {
  recognizer->impl->Reset(stream->impl.get());
}

void InputFinished(SherpaOnnxOnlineStream *stream) {
  stream->impl->InputFinished();
}

int32_t IsEndpoint(SherpaOnnxOnlineRecognizer *recognizer,
                   SherpaOnnxOnlineStream *stream) {
  return recognizer->impl->IsEndpoint(stream->impl.get());
}

SherpaOnnxDisplay *CreateDisplay(int32_t max_word_per_line) {
  SherpaOnnxDisplay *ans = new SherpaOnnxDisplay;
  ans->impl = std::make_unique<sherpa_onnx::Display>(max_word_per_line);
  return ans;
}

void DestroyDisplay(SherpaOnnxDisplay *display) { delete display; }

void SherpaOnnxPrint(SherpaOnnxDisplay *display, int32_t idx, const char *s) {
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

SherpaOnnxOfflineRecognizer *CreateOfflineRecognizer(
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
  recognizer_config.model_config.model_type =
      SHERPA_ONNX_OR(config->model_config.model_type, "");

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

  if (config->model_config.debug) {
    SHERPA_ONNX_LOGE("%s", recognizer_config.ToString().c_str());
  }

  if (!recognizer_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxOfflineRecognizer *recognizer = new SherpaOnnxOfflineRecognizer;

  recognizer->impl =
      std::make_unique<sherpa_onnx::OfflineRecognizer>(recognizer_config);

  return recognizer;
}

void DestroyOfflineRecognizer(SherpaOnnxOfflineRecognizer *recognizer) {
  delete recognizer;
}

SherpaOnnxOfflineStream *CreateOfflineStream(
    const SherpaOnnxOfflineRecognizer *recognizer) {
  SherpaOnnxOfflineStream *stream =
      new SherpaOnnxOfflineStream(recognizer->impl->CreateStream());
  return stream;
}

void DestroyOfflineStream(SherpaOnnxOfflineStream *stream) { delete stream; }

void AcceptWaveformOffline(SherpaOnnxOfflineStream *stream, int32_t sample_rate,
                           const float *samples, int32_t n) {
  stream->impl->AcceptWaveform(sample_rate, samples, n);
}

void DecodeOfflineStream(SherpaOnnxOfflineRecognizer *recognizer,
                         SherpaOnnxOfflineStream *stream) {
  recognizer->impl->DecodeStream(stream->impl.get());
}

void DecodeMultipleOfflineStreams(SherpaOnnxOfflineRecognizer *recognizer,
                                  SherpaOnnxOfflineStream **streams,
                                  int32_t n) {
  std::vector<sherpa_onnx::OfflineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  recognizer->impl->DecodeStreams(ss.data(), n);
}

const SherpaOnnxOfflineRecognizerResult *GetOfflineStreamResult(
    SherpaOnnxOfflineStream *stream) {
  const sherpa_onnx::OfflineRecognitionResult &result =
      stream->impl->GetResult();
  const auto &text = result.text;

  auto r = new SherpaOnnxOfflineRecognizerResult;
  memset(r, 0, sizeof(SherpaOnnxOfflineRecognizerResult));

  char *pText = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), pText);
  pText[text.size()] = 0;
  r->text = pText;

  if (!result.timestamps.empty()) {
    r->timestamps = new float[result.timestamps.size()];
    std::copy(result.timestamps.begin(), result.timestamps.end(),
              r->timestamps);
    r->count = result.timestamps.size();
  } else {
    r->timestamps = nullptr;
    r->count = 0;
  }

  return r;
}

void DestroyOfflineRecognizerResult(
    const SherpaOnnxOfflineRecognizerResult *r) {
  if (r) {
    delete[] r->text;
    delete[] r->timestamps;
    delete r;
  }
}

// ============================================================
// For Keyword Spot
// ============================================================

struct SherpaOnnxKeywordSpotter {
  std::unique_ptr<sherpa_onnx::KeywordSpotter> impl;
};

SherpaOnnxKeywordSpotter *CreateKeywordSpotter(
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
  spotter_config.model_config.num_threads =
      SHERPA_ONNX_OR(config->model_config.num_threads, 1);
  spotter_config.model_config.provider =
      SHERPA_ONNX_OR(config->model_config.provider, "cpu");
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

  if (config->model_config.debug) {
    SHERPA_ONNX_LOGE("%s\n", spotter_config.ToString().c_str());
  }

  if (!spotter_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config!");
    return nullptr;
  }

  SherpaOnnxKeywordSpotter *spotter = new SherpaOnnxKeywordSpotter;

  spotter->impl = std::make_unique<sherpa_onnx::KeywordSpotter>(spotter_config);

  return spotter;
}

void DestroyKeywordSpotter(SherpaOnnxKeywordSpotter *spotter) {
  delete spotter;
}

SherpaOnnxOnlineStream *CreateKeywordStream(
    const SherpaOnnxKeywordSpotter *spotter) {
  SherpaOnnxOnlineStream *stream =
      new SherpaOnnxOnlineStream(spotter->impl->CreateStream());
  return stream;
}

int32_t IsKeywordStreamReady(SherpaOnnxKeywordSpotter *spotter,
                             SherpaOnnxOnlineStream *stream) {
  return spotter->impl->IsReady(stream->impl.get());
}

void DecodeKeywordStream(SherpaOnnxKeywordSpotter *spotter,
                         SherpaOnnxOnlineStream *stream) {
  return spotter->impl->DecodeStream(stream->impl.get());
}

void DecodeMultipleKeywordStreams(SherpaOnnxKeywordSpotter *spotter,
                                  SherpaOnnxOnlineStream **streams, int32_t n) {
  std::vector<sherpa_onnx::OnlineStream *> ss(n);
  for (int32_t i = 0; i != n; ++i) {
    ss[i] = streams[i]->impl.get();
  }
  spotter->impl->DecodeStreams(ss.data(), n);
}

const SherpaOnnxKeywordResult *GetKeywordResult(
    SherpaOnnxKeywordSpotter *spotter, SherpaOnnxOnlineStream *stream) {
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
  const auto &json = result.AsJsonString();
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

void DestroyKeywordResult(const SherpaOnnxKeywordResult *r) {
  if (r) {
    delete[] r->keyword;
    delete[] r->json;
    delete[] r->tokens;
    delete[] r->tokens_arr;
    delete[] r->timestamps;
    delete r;
  }
}

// ============================================================
// For VAD
// ============================================================
//
struct SherpaOnnxCircularBuffer {
  std::unique_ptr<sherpa_onnx::CircularBuffer> impl;
};

SherpaOnnxCircularBuffer *SherpaOnnxCreateCircularBuffer(int32_t capacity) {
  SherpaOnnxCircularBuffer *buffer = new SherpaOnnxCircularBuffer;
  buffer->impl = std::make_unique<sherpa_onnx::CircularBuffer>(capacity);
  return buffer;
}

void SherpaOnnxDestroyCircularBuffer(SherpaOnnxCircularBuffer *buffer) {
  delete buffer;
}

void SherpaOnnxCircularBufferPush(SherpaOnnxCircularBuffer *buffer,
                                  const float *p, int32_t n) {
  buffer->impl->Push(p, n);
}

const float *SherpaOnnxCircularBufferGet(SherpaOnnxCircularBuffer *buffer,
                                         int32_t start_index, int32_t n) {
  std::vector<float> v = buffer->impl->Get(start_index, n);

  float *p = new float[n];
  std::copy(v.begin(), v.end(), p);
  return p;
}

void SherpaOnnxCircularBufferFree(const float *p) { delete[] p; }

void SherpaOnnxCircularBufferPop(SherpaOnnxCircularBuffer *buffer, int32_t n) {
  buffer->impl->Pop(n);
}

int32_t SherpaOnnxCircularBufferSize(SherpaOnnxCircularBuffer *buffer) {
  return buffer->impl->Size();
}

int32_t SherpaOnnxCircularBufferHead(SherpaOnnxCircularBuffer *buffer) {
  return buffer->impl->Head();
}

void SherpaOnnxCircularBufferReset(SherpaOnnxCircularBuffer *buffer) {
  buffer->impl->Reset();
}

struct SherpaOnnxVoiceActivityDetector {
  std::unique_ptr<sherpa_onnx::VoiceActivityDetector> impl;
};

SherpaOnnxVoiceActivityDetector *SherpaOnnxCreateVoiceActivityDetector(
    const SherpaOnnxVadModelConfig *config, float buffer_size_in_seconds) {
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

  vad_config.sample_rate = SHERPA_ONNX_OR(config->sample_rate, 16000);
  vad_config.num_threads = SHERPA_ONNX_OR(config->num_threads, 1);
  vad_config.provider = SHERPA_ONNX_OR(config->provider, "cpu");
  vad_config.debug = SHERPA_ONNX_OR(config->debug, false);

  if (vad_config.debug) {
    SHERPA_ONNX_LOGE("%s", vad_config.ToString().c_str());
  }

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
    SherpaOnnxVoiceActivityDetector *p) {
  delete p;
}

void SherpaOnnxVoiceActivityDetectorAcceptWaveform(
    SherpaOnnxVoiceActivityDetector *p, const float *samples, int32_t n) {
  p->impl->AcceptWaveform(samples, n);
}

int32_t SherpaOnnxVoiceActivityDetectorEmpty(
    SherpaOnnxVoiceActivityDetector *p) {
  return p->impl->Empty();
}

int32_t SherpaOnnxVoiceActivityDetectorDetected(
    SherpaOnnxVoiceActivityDetector *p) {
  return p->impl->IsSpeechDetected();
}

void SherpaOnnxVoiceActivityDetectorPop(SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Pop();
}

void SherpaOnnxVoiceActivityDetectorClear(SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Clear();
}

const SherpaOnnxSpeechSegment *SherpaOnnxVoiceActivityDetectorFront(
    SherpaOnnxVoiceActivityDetector *p) {
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

void SherpaOnnxVoiceActivityDetectorReset(SherpaOnnxVoiceActivityDetector *p) {
  p->impl->Reset();
}

#if SHERPA_ONNX_ENABLE_TTS == 1
struct SherpaOnnxOfflineTts {
  std::unique_ptr<sherpa_onnx::OfflineTts> impl;
};

SherpaOnnxOfflineTts *SherpaOnnxCreateOfflineTts(
    const SherpaOnnxOfflineTtsConfig *config) {
  sherpa_onnx::OfflineTtsConfig tts_config;

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

  tts_config.model.num_threads = SHERPA_ONNX_OR(config->model.num_threads, 1);
  tts_config.model.debug = config->model.debug;
  tts_config.model.provider = SHERPA_ONNX_OR(config->model.provider, "cpu");
  tts_config.rule_fsts = SHERPA_ONNX_OR(config->rule_fsts, "");
  tts_config.max_num_sentences = SHERPA_ONNX_OR(config->max_num_sentences, 2);

  if (tts_config.model.debug) {
    SHERPA_ONNX_LOGE("%s\n", tts_config.ToString().c_str());
  }

  if (!tts_config.Validate()) {
    SHERPA_ONNX_LOGE("Errors in config");
    return nullptr;
  }

  SherpaOnnxOfflineTts *tts = new SherpaOnnxOfflineTts;

  tts->impl = std::make_unique<sherpa_onnx::OfflineTts>(tts_config);

  return tts;
}

void SherpaOnnxDestroyOfflineTts(SherpaOnnxOfflineTts *tts) { delete tts; }

int32_t SherpaOnnxOfflineTtsSampleRate(const SherpaOnnxOfflineTts *tts) {
  return tts->impl->SampleRate();
}

int32_t SherpaOnnxOfflineTtsNumSpeakers(const SherpaOnnxOfflineTts *tts) {
  return tts->impl->NumSpeakers();
}

static const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateInternal(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid,
    float speed, std::function<void(const float *, int32_t, float)> callback) 
{
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
  return SherpaOnnxOfflineTtsGenerateInternal( tts, text, sid, speed, nullptr );
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateWithCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallback callback) {
  auto wrapper = [callback](const float *samples, int32_t n, float /*progress*/) {
    callback(samples, n );
  };

  return SherpaOnnxOfflineTtsGenerateInternal( tts, text, sid, speed, wrapper );
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateWithProgressCallback(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioProgressCallback callback) {
  auto wrapper = [callback](const float *samples, int32_t n, float progress) {
    callback(samples, n, progress );
  };
  return SherpaOnnxOfflineTtsGenerateInternal( tts, text, sid, speed, wrapper );
}

const SherpaOnnxGeneratedAudio *SherpaOnnxOfflineTtsGenerateWithCallbackWithArg(
    const SherpaOnnxOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaOnnxGeneratedAudioCallbackWithArg callback, void *arg) {
  auto wrapper = [callback, arg](const float *samples, int32_t n, float /*progress*/) {
    callback(samples, n, arg);
  };

  return SherpaOnnxOfflineTtsGenerateInternal( tts, text, sid, speed, wrapper );
}

void SherpaOnnxDestroyOfflineTtsGeneratedAudio(
    const SherpaOnnxGeneratedAudio *p) {
  if (p) {
    delete[] p->samples;
    delete p;
  }
}
#endif  // SHERPA_ONNX_ENABLE_TTS == 1

int32_t SherpaOnnxWriteWave(const float *samples, int32_t n,
                            int32_t sample_rate, const char *filename) {
  return sherpa_onnx::WriteWave(filename, sample_rate, samples, n);
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
