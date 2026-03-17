// kws_api.cc - KWS 封装实现，依赖 sherpa-onnx CXX API
#include "kws_api.h"
#include "sherpa-onnx/c-api/cxx-api.h"
#include <memory>

namespace kws {

namespace so = sherpa_onnx::cxx;

struct Engine {
  so::KeywordSpotter spotter;
  explicit Engine(so::KeywordSpotter s) : spotter(std::move(s)) {}
};

struct Stream {
  so::OnlineStream stream;
  explicit Stream(so::OnlineStream s) : stream(std::move(s)) {}
};

static so::KeywordSpotterConfig ToSpotterConfig(const Config& c) {
  so::KeywordSpotterConfig config;
  config.feat_config.sample_rate = c.sample_rate;
  config.feat_config.feature_dim = 80;
  config.model_config.transducer.encoder = c.encoder_path;
  config.model_config.transducer.decoder = c.decoder_path;
  config.model_config.transducer.joiner = c.joiner_path;
  config.model_config.tokens = c.tokens_path;
  config.model_config.num_threads = c.num_threads;
  config.model_config.provider = "cpu";
  config.model_config.debug = false;
  config.keywords_file = c.keywords_file_path;
  config.keywords_score = c.keywords_score;
  config.keywords_threshold = c.keywords_threshold;
  config.num_trailing_blanks = c.num_trailing_blanks;
  config.max_active_paths = c.max_active_paths;
  return config;
}

Engine* Create(const Config& config) {
  so::KeywordSpotterConfig sc = ToSpotterConfig(config);
  so::KeywordSpotter spotter = so::KeywordSpotter::Create(sc);
  if (!spotter.Get()) return nullptr;
  return new Engine(std::move(spotter));
}

void Destroy(Engine* engine) {
  delete engine;
}

Stream* CreateStream(Engine* engine) {
  if (!engine) return nullptr;
  return new Stream(engine->spotter.CreateStream());
}

void DestroyStream(Stream* stream) {
  delete stream;
}

void AcceptWaveform(Engine* engine, Stream* stream, int sample_rate,
                    const float* data, int num_samples) {
  if (!engine || !stream || !data || num_samples <= 0) return;
  stream->stream.AcceptWaveform(sample_rate, data, num_samples);
}

bool IsReady(Engine* engine, Stream* stream) {
  if (!engine || !stream) return false;
  return engine->spotter.IsReady(&stream->stream);
}

void Decode(Engine* engine, Stream* stream) {
  if (!engine || !stream) return;
  engine->spotter.Decode(&stream->stream);
}

Result GetResult(Engine* engine, Stream* stream) {
  Result r;
  if (!engine || !stream) return r;
  auto sr = engine->spotter.GetResult(&stream->stream);
  r.keyword = sr.keyword;
  r.full_tokens = sr.full_tokens;
  r.timestamps = sr.timestamps;
  r.start_time = sr.start_time;
  return r;
}

void Reset(Engine* engine, Stream* stream) {
  if (!engine || !stream) return;
  engine->spotter.Reset(&stream->stream);
}

}  // namespace kws
