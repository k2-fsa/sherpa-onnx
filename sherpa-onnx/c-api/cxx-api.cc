// sherpa-onnx/c-api/cxx-api.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-onnx/c-api/cxx-api.h"

#include <algorithm>
#include <cstring>

namespace sherpa_onnx::cxx {

Wave ReadWave(const std::string &filename) {
  auto p = SherpaOnnxReadWave(filename.c_str());

  Wave ans;
  if (p) {
    ans.samples.resize(p->num_samples);

    std::copy(p->samples, p->samples + p->num_samples, ans.samples.data());

    ans.sample_rate = p->sample_rate;
    SherpaOnnxFreeWave(p);
  }

  return ans;
}

OnlineStream::OnlineStream(const SherpaOnnxOnlineStream *p)
    : MoveOnly<OnlineStream, SherpaOnnxOnlineStream>(p) {}

void OnlineStream::Destroy(const SherpaOnnxOnlineStream *p) const {
  SherpaOnnxDestroyOnlineStream(p);
}

void OnlineStream::AcceptWaveform(int32_t sample_rate, const float *samples,
                                  int32_t n) const {
  SherpaOnnxOnlineStreamAcceptWaveform(p_, sample_rate, samples, n);
}

OnlineRecognizer OnlineRecognizer::Create(
    const OnlineRecognizerConfig &config) {
  struct SherpaOnnxOnlineRecognizerConfig c;
  memset(&c, 0, sizeof(c));

  c.feat_config.sample_rate = config.feat_config.sample_rate;
  c.feat_config.feature_dim = config.feat_config.feature_dim;

  c.model_config.transducer.encoder =
      config.model_config.transducer.encoder.c_str();
  c.model_config.transducer.decoder =
      config.model_config.transducer.decoder.c_str();
  c.model_config.transducer.joiner =
      config.model_config.transducer.joiner.c_str();

  c.model_config.paraformer.encoder =
      config.model_config.paraformer.encoder.c_str();
  c.model_config.paraformer.decoder =
      config.model_config.paraformer.decoder.c_str();

  c.model_config.zipformer2_ctc.model =
      config.model_config.zipformer2_ctc.model.c_str();

  c.model_config.tokens = config.model_config.tokens.c_str();
  c.model_config.num_threads = config.model_config.num_threads;
  c.model_config.provider = config.model_config.provider.c_str();
  c.model_config.debug = config.model_config.debug;
  c.model_config.model_type = config.model_config.model_type.c_str();
  c.model_config.modeling_unit = config.model_config.modeling_unit.c_str();
  c.model_config.bpe_vocab = config.model_config.bpe_vocab.c_str();
  c.model_config.tokens_buf = config.model_config.tokens_buf.c_str();
  c.model_config.tokens_buf_size = config.model_config.tokens_buf.size();

  c.decoding_method = config.decoding_method.c_str();
  c.max_active_paths = config.max_active_paths;
  c.enable_endpoint = config.enable_endpoint;
  c.rule1_min_trailing_silence = config.rule1_min_trailing_silence;
  c.rule2_min_trailing_silence = config.rule2_min_trailing_silence;
  c.rule3_min_utterance_length = config.rule3_min_utterance_length;
  c.hotwords_file = config.hotwords_file.c_str();
  c.hotwords_score = config.hotwords_score;

  c.ctc_fst_decoder_config.graph = config.ctc_fst_decoder_config.graph.c_str();
  c.ctc_fst_decoder_config.max_active =
      config.ctc_fst_decoder_config.max_active;

  c.rule_fsts = config.rule_fsts.c_str();
  c.rule_fars = config.rule_fars.c_str();

  c.blank_penalty = config.blank_penalty;

  c.hotwords_buf = config.hotwords_buf.c_str();
  c.hotwords_buf_size = config.hotwords_buf.size();

  auto p = SherpaOnnxCreateOnlineRecognizer(&c);
  return OnlineRecognizer(p);
}

OnlineRecognizer::OnlineRecognizer(const SherpaOnnxOnlineRecognizer *p)
    : MoveOnly<OnlineRecognizer, SherpaOnnxOnlineRecognizer>(p) {}

void OnlineRecognizer::Destroy(const SherpaOnnxOnlineRecognizer *p) const {
  SherpaOnnxDestroyOnlineRecognizer(p);
}

OnlineStream OnlineRecognizer::CreateStream() const {
  auto s = SherpaOnnxCreateOnlineStream(p_);
  return OnlineStream{s};
}

OnlineStream OnlineRecognizer::CreateStream(const std::string &hotwords) const {
  auto s = SherpaOnnxCreateOnlineStreamWithHotwords(p_, hotwords.c_str());
  return OnlineStream{s};
}

bool OnlineRecognizer::IsReady(const OnlineStream *s) const {
  return SherpaOnnxIsOnlineStreamReady(p_, s->Get());
}

void OnlineRecognizer::Decode(const OnlineStream *s) const {
  SherpaOnnxDecodeOnlineStream(p_, s->Get());
}

void OnlineRecognizer::Decode(const OnlineStream *ss, int32_t n) const {
  if (n <= 0) {
    return;
  }

  std::vector<const SherpaOnnxOnlineStream *> streams(n);
  for (int32_t i = 0; i != n; ++n) {
    streams[i] = ss[i].Get();
  }

  SherpaOnnxDecodeMultipleOnlineStreams(p_, streams.data(), n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(
    const OnlineStream *s) const {
  auto r = SherpaOnnxGetOnlineStreamResult(p_, s->Get());

  OnlineRecognizerResult ans;
  ans.text = r->text;

  ans.tokens.resize(r->count);
  for (int32_t i = 0; i != r->count; ++i) {
    ans.tokens[i] = r->tokens_arr[i];
  }

  if (r->timestamps) {
    ans.timestamps.resize(r->count);
    std::copy(r->timestamps, r->timestamps + r->count, ans.timestamps.data());
  }

  ans.json = r->json;

  SherpaOnnxDestroyOnlineRecognizerResult(r);

  return ans;
}

}  // namespace sherpa_onnx::cxx
