// sherpa-onnx/csrc/sherpa-onnx.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  if (argc < 6 || argc > 8) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-onnx \
    /path/to/tokens.txt \
    /path/to/encoder.onnx \
    /path/to/decoder.onnx \
    /path/to/joiner.onnx \
    /path/to/foo.wav [num_threads [decoding_method]]

Default value for num_threads is 2.
Valid values for decoding_method: greedy_search (default), modified_beam_search.
foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";
    fprintf(stderr, "%s\n", usage);

    return 0;
  }

  sherpa_onnx::OnlineRecognizerConfig config;

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
  config.max_active_paths = 4;

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  sherpa_onnx::OnlineRecognizer recognizer(config);

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

  fprintf(stderr, "wav filename: %s\n", wav_filename.c_str());
  fprintf(stderr, "wav duration (s): %.3f\n", duration);

  auto begin = std::chrono::steady_clock::now();
  fprintf(stderr, "Started\n");

  auto s = recognizer.CreateStream();
  s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

  std::vector<float> tail_paddings(static_cast<int>(0.2 * sampling_rate));
  // Note: We can call AcceptWaveform() multiple times.
  s->AcceptWaveform(sampling_rate, tail_paddings.data(), tail_paddings.size());

  // Call InputFinished() to indicate that no audio samples are available
  s->InputFinished();

  while (recognizer.IsReady(s.get())) {
    recognizer.DecodeStream(s.get());
  }

  std::string text = recognizer.GetResult(s.get()).AsJsonString();

  fprintf(stderr, "Done!\n");

  fprintf(stderr, "Recognition result for %s:\n%s\n", wav_filename.c_str(),
          text.c_str());

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
