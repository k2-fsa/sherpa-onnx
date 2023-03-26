// sherpa-onnx/csrc/sherpa-onnx-offline.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-model.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  if (argc < 6 || argc > 8) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-onnx-offline \
    /path/to/tokens.txt \
    /path/to/encoder.onnx \
    /path/to/decoder.onnx \
    /path/to/joiner.onnx \
    /path/to/foo.wav [num_threads [decoding_method]]

Default value for num_threads is 2.
Valid values for decoding_method: greedy_search.
foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";
    fprintf(stderr, "%s\n", usage);

    return 0;
  }

  sherpa_onnx::OfflineRecognizerConfig config;

  config.model_config.tokens = argv[1];

  config.model_config.debug = false;
  config.model_config.encoder_filename = argv[2];
  config.model_config.decoder_filename = argv[3];
  config.model_config.joiner_filename = argv[4];

  std::string wav_filename = argv[5];

  config.model_config.num_threads = 2;
  if (argc == 7 && atoi(argv[6]) > 0) {
    config.model_config.num_threads = atoi(argv[6]);
  }

  if (argc == 8) {
    config.decoding_method = argv[7];
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  int32_t sampling_rate = -1;

  bool is_ok = false;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
    return -1;
  }
  fprintf(stderr, "sampling rate of input file: %d\n", sampling_rate);

  float duration = samples.size() / static_cast<float>(sampling_rate);

  sherpa_onnx::OfflineRecognizer recognizer(config);
  auto s = recognizer.CreateStream();

  auto begin = std::chrono::steady_clock::now();
  fprintf(stderr, "Started\n");

  s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

  recognizer.DecodeStream(s.get());

  fprintf(stderr, "Done!\n");

  fprintf(stderr, "Recognition result for %s:\n%s\n", wav_filename.c_str(),
          s->GetResult().text.c_str());

  auto end = std::chrono::steady_clock::now();
  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "num threads: %d\n", config.model_config.num_threads);
  fprintf(stderr, "decoding method: %s\n", config.decoding_method.c_str());

  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
