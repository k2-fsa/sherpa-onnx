// cxx-api-examples/speaker-identification-cxx-api.cc
//
// Copyright (c)  2026  Xiaomi Corporation

// We assume you have pre-downloaded the speaker embedding extractor model
// from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
//
// An example command to download
// "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
// is given below:
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
//
// clang-format on
//
// Also, please download the test wave files from
//
// https://github.com/csukuangfj/sr-data

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sherpa-onnx/c-api/cxx-api.h"

using namespace sherpa_onnx::cxx;  // NOLINT

// RAII wrapper for a speaker embedding (float array returned by the C API).
class EmbeddingGuard {
 public:
  EmbeddingGuard() = default;
  explicit EmbeddingGuard(const float *p) : p_(p) {}
  ~EmbeddingGuard() {
    if (p_) {
      SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(p_);
    }
  }

  // non-copyable
  EmbeddingGuard(const EmbeddingGuard &) = delete;
  EmbeddingGuard &operator=(const EmbeddingGuard &) = delete;

  // movable
  EmbeddingGuard(EmbeddingGuard &&o) noexcept : p_(o.p_) { o.p_ = nullptr; }
  EmbeddingGuard &operator=(EmbeddingGuard &&o) noexcept {
    if (this != &o) {
      if (p_) SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(p_);
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }

  const float *Get() const { return p_; }
  explicit operator bool() const { return p_ != nullptr; }

 private:
  const float *p_ = nullptr;
};

static EmbeddingGuard ComputeEmbedding(
    const SherpaOnnxSpeakerEmbeddingExtractor *ex,
    const std::string &wav_filename) {
  Wave wave = ReadWave(wav_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read " << wav_filename << "\n";
    exit(-1);
  }

  const SherpaOnnxOnlineStream *stream =
      SherpaOnnxSpeakerEmbeddingExtractorCreateStream(ex);

  SherpaOnnxOnlineStreamAcceptWaveform(stream, wave.sample_rate,
                                       wave.samples.data(),
                                       wave.samples.size());
  SherpaOnnxOnlineStreamInputFinished(stream);

  if (!SherpaOnnxSpeakerEmbeddingExtractorIsReady(ex, stream)) {
    std::cerr << "The input wave file " << wav_filename << " is too short!\n";
    SherpaOnnxDestroyOnlineStream(stream);
    exit(-1);
  }

  // we will free the embedding via EmbeddingGuard
  const float *v =
      SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(ex, stream);

  SherpaOnnxDestroyOnlineStream(stream);

  return EmbeddingGuard(v);
}

int32_t main() {
  SherpaOnnxSpeakerEmbeddingExtractorConfig config;

  memset(&config, 0, sizeof(config));

  // please download the model from
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
  config.model = "./3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";

  config.num_threads = 1;
  config.debug = 0;
  config.provider = "cpu";

  const SherpaOnnxSpeakerEmbeddingExtractor *ex =
      SherpaOnnxCreateSpeakerEmbeddingExtractor(&config);
  if (!ex) {
    std::cerr << "Failed to create speaker embedding extractor\n";
    return -1;
  }

  int32_t dim = SherpaOnnxSpeakerEmbeddingExtractorDim(ex);

  const SherpaOnnxSpeakerEmbeddingManager *manager =
      SherpaOnnxCreateSpeakerEmbeddingManager(dim);

  // Please download the test data from
  // https://github.com/csukuangfj/sr-data
  std::string spk1_1 = "./sr-data/enroll/fangjun-sr-1.wav";
  std::string spk1_2 = "./sr-data/enroll/fangjun-sr-2.wav";
  std::string spk1_3 = "./sr-data/enroll/fangjun-sr-3.wav";

  std::string spk2_1 = "./sr-data/enroll/leijun-sr-1.wav";
  std::string spk2_2 = "./sr-data/enroll/leijun-sr-2.wav";

  // Store embeddings in RAII wrappers so they are freed automatically.
  std::vector<EmbeddingGuard> spk1_guards;
  spk1_guards.push_back(ComputeEmbedding(ex, spk1_1));
  spk1_guards.push_back(ComputeEmbedding(ex, spk1_2));
  spk1_guards.push_back(ComputeEmbedding(ex, spk1_3));

  std::vector<EmbeddingGuard> spk2_guards;
  spk2_guards.push_back(ComputeEmbedding(ex, spk2_1));
  spk2_guards.push_back(ComputeEmbedding(ex, spk2_2));

  const float *spk1_vec[4] = {nullptr};
  spk1_vec[0] = spk1_guards[0].Get();
  spk1_vec[1] = spk1_guards[1].Get();
  spk1_vec[2] = spk1_guards[2].Get();

  const float *spk2_vec[3] = {nullptr};
  spk2_vec[0] = spk2_guards[0].Get();
  spk2_vec[1] = spk2_guards[1].Get();

  if (!SherpaOnnxSpeakerEmbeddingManagerAddList(manager, "fangjun",
                                                spk1_vec)) {
    std::cerr << "Failed to register fangjun\n";
    exit(-1);
  }

  if (!SherpaOnnxSpeakerEmbeddingManagerContains(manager, "fangjun")) {
    std::cerr << "Failed to find fangjun\n";
    exit(-1);
  }

  if (!SherpaOnnxSpeakerEmbeddingManagerAddList(manager, "leijun",
                                                spk2_vec)) {
    std::cerr << "Failed to register leijun\n";
    exit(-1);
  }

  if (!SherpaOnnxSpeakerEmbeddingManagerContains(manager, "leijun")) {
    std::cerr << "Failed to find leijun\n";
    exit(-1);
  }

  if (SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(manager) != 2) {
    std::cerr << "There should be two speakers: fangjun and leijun\n";
    exit(-1);
  }

  const char *const *all_speakers =
      SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(manager);
  const char *const *p = all_speakers;
  std::cerr << "list of registered speakers\n-----\n";
  while (p[0]) {
    std::cerr << "speaker: " << p[0] << "\n";
    ++p;
  }
  std::cerr << "----\n";

  SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(all_speakers);

  std::string test1 = "./sr-data/test/fangjun-test-sr-1.wav";
  std::string test2 = "./sr-data/test/leijun-test-sr-1.wav";
  std::string test3 = "./sr-data/test/liudehua-test-sr-1.wav";

  EmbeddingGuard v1 = ComputeEmbedding(ex, test1);
  EmbeddingGuard v2 = ComputeEmbedding(ex, test2);
  EmbeddingGuard v3 = ComputeEmbedding(ex, test3);

  float threshold = 0.6;

  const char *name1 =
      SherpaOnnxSpeakerEmbeddingManagerSearch(manager, v1.Get(), threshold);
  if (name1) {
    std::cerr << test1 << ": Found " << name1 << "\n";
    SherpaOnnxSpeakerEmbeddingManagerFreeSearch(name1);
  } else {
    std::cerr << test1 << ": Not found\n";
  }

  const char *name2 =
      SherpaOnnxSpeakerEmbeddingManagerSearch(manager, v2.Get(), threshold);
  if (name2) {
    std::cerr << test2 << ": Found " << name2 << "\n";
    SherpaOnnxSpeakerEmbeddingManagerFreeSearch(name2);
  } else {
    std::cerr << test2 << ": Not found\n";
  }

  const char *name3 =
      SherpaOnnxSpeakerEmbeddingManagerSearch(manager, v3.Get(), threshold);
  if (name3) {
    std::cerr << test3 << ": Found " << name3 << "\n";
    SherpaOnnxSpeakerEmbeddingManagerFreeSearch(name3);
  } else {
    std::cerr << test3 << ": Not found\n";
  }

  int32_t ok = SherpaOnnxSpeakerEmbeddingManagerVerify(manager, "fangjun",
                                                        v1.Get(), threshold);
  if (ok) {
    std::cerr << test1 << " matches fangjun\n";
  } else {
    std::cerr << test1 << " does NOT match fangjun\n";
  }

  ok = SherpaOnnxSpeakerEmbeddingManagerVerify(manager, "fangjun", v2.Get(),
                                                threshold);
  if (ok) {
    std::cerr << test2 << " matches fangjun\n";
  } else {
    std::cerr << test2 << " does NOT match fangjun\n";
  }

  std::cerr << "Removing fangjun\n";
  if (!SherpaOnnxSpeakerEmbeddingManagerRemove(manager, "fangjun")) {
    std::cerr << "Failed to remove fangjun\n";
    exit(-1);
  }

  if (SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(manager) != 1) {
    std::cerr << "There should be only 1 speaker left\n";
    exit(-1);
  }

  name1 = SherpaOnnxSpeakerEmbeddingManagerSearch(manager, v1.Get(), threshold);
  if (name1) {
    std::cerr << test1 << ": Found " << name1 << "\n";
    SherpaOnnxSpeakerEmbeddingManagerFreeSearch(name1);
  } else {
    std::cerr << test1 << ": Not found\n";
  }

  std::cerr << "Removing leijun\n";
  if (!SherpaOnnxSpeakerEmbeddingManagerRemove(manager, "leijun")) {
    std::cerr << "Failed to remove leijun\n";
    exit(-1);
  }

  if (SherpaOnnxSpeakerEmbeddingManagerNumSpeakers(manager) != 0) {
    std::cerr << "There should be only 0 speakers left\n";
    exit(-1);
  }

  name2 = SherpaOnnxSpeakerEmbeddingManagerSearch(manager, v2.Get(), threshold);
  if (name2) {
    std::cerr << test2 << ": Found " << name2 << "\n";
    SherpaOnnxSpeakerEmbeddingManagerFreeSearch(name2);
  } else {
    std::cerr << test2 << ": Not found\n";
  }

  all_speakers = SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers(manager);

  p = all_speakers;
  std::cerr << "list of registered speakers\n-----\n";
  while (p[0]) {
    std::cerr << "speaker: " << p[0] << "\n";
    ++p;
  }
  std::cerr << "----\n";

  SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers(all_speakers);

  SherpaOnnxDestroySpeakerEmbeddingManager(manager);
  SherpaOnnxDestroySpeakerEmbeddingExtractor(ex);

  return 0;
}
