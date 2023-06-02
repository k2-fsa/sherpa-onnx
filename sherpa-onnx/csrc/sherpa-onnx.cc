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
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Usage:

  ./bin/sherpa-onnx \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --provider=cpu \
    --num-threads=2 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]

Note: It supports decoding multiple files in batches

Default value for num_threads is 2.
Valid values for decoding_method: greedy_search (default), modified_beam_search.
Valid values for provider: cpu (default), cuda, coreml.
foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OnlineRecognizerConfig config;

  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  sherpa_onnx::OnlineRecognizer recognizer(config);

  float duration = 0;
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    const std::string wav_filename = po.GetArg(i);
    int32_t sampling_rate = -1;

    bool is_ok = false;
    const std::vector<float> samples =
        sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);

    if (!is_ok) {
      fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
      return -1;
    }
    fprintf(stderr, "sampling rate of input file: %d\n", sampling_rate);

    const float duration = samples.size() / static_cast<float>(sampling_rate);

    fprintf(stderr, "wav filename: %s\n", wav_filename.c_str());
    fprintf(stderr, "wav duration (s): %.3f\n", duration);

    fprintf(stderr, "Started\n");
    const auto begin = std::chrono::steady_clock::now();

    auto s = recognizer.CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

    std::vector<float> tail_paddings(static_cast<int>(0.3 * sampling_rate));
    // Note: We can call AcceptWaveform() multiple times.
    s->AcceptWaveform(
      sampling_rate, tail_paddings.data(), tail_paddings.size());

    // Call InputFinished() to indicate that no audio samples are available
    s->InputFinished();

    while (recognizer.IsReady(s.get())) {
      recognizer.DecodeStream(s.get());
    }

    const std::string text = recognizer.GetResult(s.get()).AsJsonString();

    const auto end = std::chrono::steady_clock::now();
    const float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() / 1000.;

    fprintf(stderr, "Done!\n");
    fprintf(stderr,
            "Recognition result for %s:\n%s\n",
            wav_filename.c_str(), text.c_str());
    fprintf(stderr, "num threads: %d\n", config.model_config.num_threads);
    fprintf(stderr, "decoding method: %s\n", config.decoding_method.c_str());
    if (config.decoding_method == "modified_beam_search") {
      fprintf(stderr, "max active paths: %d\n", config.max_active_paths);
    }

    fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
    const float rtf = elapsed_seconds / duration;
    fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
            elapsed_seconds, duration, rtf);
  }

  return 0;
}
