// sherpa-onnx/csrc/sherpa-onnx-offline-speaker-diarization.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-speaker-diarization.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

static int32_t ProgressCallback(int32_t processed_chunks, int32_t num_chunks,
                                void *arg) {
  float progress = 100.0 * processed_chunks / num_chunks;
  fprintf(stderr, "progress %.2f%%\n", progress);

  // the return value is currently ignored
  return 0;
}

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Offline/Non-streaming speaker diarization with sherpa-onnx
Usage example:

  )usage";
  sherpa_onnx::OfflineSpeakerDiarizationConfig config;
  sherpa_onnx::ParseOptions po(kUsageMessage);
  config.Register(&po);
  po.Read(argc, argv);

  std::cout << config.ToString() << "\n";

  if (!config.Validate()) {
    po.PrintUsage();
    std::cerr << "Errors in config!\n";
    return -1;
  }

  if (po.NumArgs() != 1) {
    std::cerr << "Error: Please provide exactly 1 wave file.\n\n";
    po.PrintUsage();
    return -1;
  }

  sherpa_onnx::OfflineSpeakerDiarization sd(config);

  std::cout << "Started\n";
  const auto begin = std::chrono::steady_clock::now();
  const std::string wav_filename = po.GetArg(1);
  int32_t sample_rate = -1;
  bool is_ok = false;
  const std::vector<float> samples =
      sherpa_onnx::ReadWave(wav_filename, &sample_rate, &is_ok);
  if (!is_ok) {
    std::cerr << "Failed to read " << wav_filename.c_str() << "\n";
    return -1;
  }

  if (sample_rate != 16000) {
    std::cerr << "Expect sample rate 16000. Given: " << sample_rate << "\n";
    return -1;
  }

  float duration = samples.size() / static_cast<float>(sample_rate);

  // sd.Process(samples.data(), samples.size() < 160000 ? samples.size() :
  // 160000);
  auto result =
      sd.Process(samples.data(), samples.size(), ProgressCallback, nullptr);

  for (const auto &r : result.segments_) {
    std::cout << r.ToString() << "\n";
  }

  const auto end = std::chrono::steady_clock::now();
  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "Duration : %.3f s\n", duration);
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
