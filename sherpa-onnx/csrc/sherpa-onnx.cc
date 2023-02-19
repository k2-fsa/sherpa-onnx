// sherpa-onnx/csrc/sherpa-onnx.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "sherpa-onnx/csrc/decode.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  if (argc < 6 || argc > 7) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-onnx \
    /path/to/tokens.txt \
    /path/to/encoder.onnx \
    /path/to/decoder.onnx \
    /path/to/joiner.onnx \
    /path/to/foo.wav [num_threads]

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";
    fprintf(stderr, "%s\n", usage);

    return 0;
  }

  std::string tokens = argv[1];
  sherpa_onnx::OnlineTransducerModelConfig config;
  config.debug = false;
  config.encoder_filename = argv[2];
  config.decoder_filename = argv[3];
  config.joiner_filename = argv[4];
  std::string wav_filename = argv[5];

  config.num_threads = 2;
  if (argc == 7) {
    config.num_threads = atoi(argv[6]);
  }
  fprintf(stderr, "%s\n", config.ToString().c_str());

  auto model = sherpa_onnx::OnlineTransducerModel::Create(config);

  sherpa_onnx::SymbolTable sym(tokens);

  Ort::AllocatorWithDefaultOptions allocator;

  int32_t chunk_size = model->ChunkSize();
  int32_t chunk_shift = model->ChunkShift();

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  std::vector<Ort::Value> states = model->GetEncoderInitStates();

  std::vector<int64_t> hyp(model->ContextSize(), 0);

  int32_t expected_sampling_rate = 16000;

  bool is_ok = false;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(wav_filename, expected_sampling_rate, &is_ok);

  if (!is_ok) {
    fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
    return -1;
  }

  float duration = samples.size() / static_cast<float>(expected_sampling_rate);

  fprintf(stderr, "wav filename: %s\n", wav_filename.c_str());
  fprintf(stderr, "wav duration (s): %.3f\n", duration);

  auto begin = std::chrono::steady_clock::now();
  fprintf(stderr, "Started\n");

  sherpa_onnx::FeatureExtractor feat_extractor;
  feat_extractor.AcceptWaveform(expected_sampling_rate, samples.data(),
                                samples.size());

  std::vector<float> tail_paddings(
      static_cast<int>(0.2 * expected_sampling_rate));
  feat_extractor.AcceptWaveform(expected_sampling_rate, tail_paddings.data(),
                                tail_paddings.size());
  feat_extractor.InputFinished();

  int32_t num_frames = feat_extractor.NumFramesReady();
  int32_t feature_dim = feat_extractor.FeatureDim();

  std::array<int64_t, 3> x_shape{1, chunk_size, feature_dim};

  for (int32_t start = 0; start + chunk_size < num_frames;
       start += chunk_shift) {
    std::vector<float> features = feat_extractor.GetFrames(start, chunk_size);

    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, features.data(), features.size(),
                                 x_shape.data(), x_shape.size());
    auto pair = model->RunEncoder(std::move(x), states);
    states = std::move(pair.second);
    sherpa_onnx::GreedySearch(model.get(), std::move(pair.first), &hyp);
  }
  std::string text;
  for (size_t i = model->ContextSize(); i != hyp.size(); ++i) {
    text += sym[hyp[i]];
  }

  fprintf(stderr, "Done!\n");

  fprintf(stderr, "Recognition result for %s:\n%s\n", wav_filename.c_str(),
          text.c_str());

  auto end = std::chrono::steady_clock::now();
  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "num threads: %d\n", config.num_threads);

  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
