// cxx-api-examples/funasr-nano-cxx-api.cc
//
// This file demonstrates how to use FunASR-nano with sherpa-onnx's C++ API.
//
// clang-format off
//
// Example usage:
//   ./bin/funasr-nano-cxx-api \
//     --funasr-nano-encoder-adaptor=/path/to/encoder_adaptor.onnx \
//     --funasr-nano-llm-prefill=/path/to/llm_prefill.onnx \
//     --funasr-nano-llm-decode=/path/to/llm_decode.onnx \
//     --funasr-nano-embedding=/path/to/embedding.onnx \
//     --funasr-nano-tokenizer=/path/to/Qwen3-0.6B \
//     /path/to/audio.wav
//
// clang-format on

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main(int32_t argc, char *argv[]) {
  using namespace sherpa_onnx::cxx;

  const char *kUsageMessage = R"usage(
FunASR-nano speech recognition example using sherpa-onnx C++ API.

Usage:
  ./bin/funasr-nano-cxx-api \
    --funasr-nano-encoder-adaptor=/path/to/encoder_adaptor.onnx \
    --funasr-nano-llm-prefill=/path/to/llm_prefill.onnx \
    --funasr-nano-llm-decode=/path/to/llm_decode.onnx \
    --funasr-nano-tokenizer=/path/to/Qwen3-0.6B \
    --funasr-nano-embedding=/path/to/embedding.onnx \
    [--funasr-nano-user-prompt="语音转写："] \
    [--funasr-nano-max-new-tokens=512] \
    [--funasr-nano-temperature=0.3] \
    [--funasr-nano-top-p=0.8] \
    /path/to/audio.wav

Required arguments:
  --funasr-nano-encoder-adaptor: Path to encoder_adaptor.onnx
  --funasr-nano-llm-prefill: Path to llm_prefill.onnx
  --funasr-nano-llm-decode: Path to llm_decode.onnx
  --funasr-nano-tokenizer: Path to tokenizer directory (e.g., Qwen3-0.6B)

Optional arguments:
  --funasr-nano-embedding: Path to embedding.onnx (recommended)
  --funasr-nano-user-prompt: User prompt template (default: "语音转写：")
  --funasr-nano-max-new-tokens: Maximum tokens to generate (default: 512)
  --funasr-nano-temperature: Sampling temperature (default: 0.3)
  --funasr-nano-top-p: Top-p sampling threshold (default: 0.8)
  --num-threads: Number of threads (default: 2)
  --provider: cpu (default) or cuda

Example:
  ./bin/funasr-nano-cxx-api \
    --funasr-nano-encoder-adaptor=./models/encoder_adaptor.onnx \
    --funasr-nano-llm-prefill=./models/llm_prefill.onnx \
    --funasr-nano-llm-decode=./models/llm_decode.onnx \
    --funasr-nano-tokenizer=./models/Qwen3-0.6B \
    --funasr-nano-embedding=./models/embedding.onnx \
    ./test.wav
)usage";

  if (argc < 5) {
    std::cerr << kUsageMessage << "\n";
    return -1;
  }

  OfflineRecognizerConfig config;
  config.model_config.num_threads = 2;
  config.model_config.debug = false;
  config.model_config.provider = "cpu";

  // Parse command line arguments
  for (int32_t i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--funasr-nano-encoder-adaptor=") == 0) {
      config.model_config.funasr_nano.encoder_adaptor =
          arg.substr(strlen("--funasr-nano-encoder-adaptor="));
    } else if (arg.find("--funasr-nano-llm-prefill=") == 0) {
      config.model_config.funasr_nano.llm_prefill =
          arg.substr(strlen("--funasr-nano-llm-prefill="));
    } else if (arg.find("--funasr-nano-llm-decode=") == 0) {
      config.model_config.funasr_nano.llm_decode =
          arg.substr(strlen("--funasr-nano-llm-decode="));
    } else if (arg.find("--funasr-nano-embedding=") == 0) {
      config.model_config.funasr_nano.embedding =
          arg.substr(strlen("--funasr-nano-embedding="));
    } else if (arg.find("--funasr-nano-tokenizer=") == 0) {
      config.model_config.funasr_nano.tokenizer =
          arg.substr(strlen("--funasr-nano-tokenizer="));
    } else if (arg.find("--funasr-nano-user-prompt=") == 0) {
      config.model_config.funasr_nano.user_prompt =
          arg.substr(strlen("--funasr-nano-user-prompt="));
    } else if (arg.find("--funasr-nano-max-new-tokens=") == 0) {
      config.model_config.funasr_nano.max_new_tokens =
          std::stoi(arg.substr(strlen("--funasr-nano-max-new-tokens=")));
    } else if (arg.find("--funasr-nano-temperature=") == 0) {
      config.model_config.funasr_nano.temperature =
          std::stof(arg.substr(strlen("--funasr-nano-temperature=")));
    } else if (arg.find("--funasr-nano-top-p=") == 0) {
      config.model_config.funasr_nano.top_p =
          std::stof(arg.substr(strlen("--funasr-nano-top-p=")));
    } else if (arg.find("--num-threads=") == 0) {
      config.model_config.num_threads =
          std::stoi(arg.substr(strlen("--num-threads=")));
    } else if (arg.find("--provider=") == 0) {
      config.model_config.provider = arg.substr(strlen("--provider="));
    } else if (arg[0] != '-') {
      // This should be the audio file
      std::string wave_filename = arg;

      std::cout << "Loading model...\n";
      std::cout << "  encoder_adaptor: "
                << config.model_config.funasr_nano.encoder_adaptor << "\n";
      std::cout << "  llm_prefill: "
                << config.model_config.funasr_nano.llm_prefill << "\n";
      std::cout << "  llm_decode: "
                << config.model_config.funasr_nano.llm_decode << "\n";
      std::cout << "  tokenizer: " << config.model_config.funasr_nano.tokenizer
                << "\n";
      if (!config.model_config.funasr_nano.embedding.empty()) {
        std::cout << "  embedding: "
                  << config.model_config.funasr_nano.embedding << "\n";
      }

      const auto begin_init = std::chrono::steady_clock::now();

      OfflineRecognizer recognizer = OfflineRecognizer::Create(config);
      if (!recognizer.Get()) {
        std::cerr << "Failed to create recognizer. Please check your config.\n";
        return -1;
      }

      const auto end_init = std::chrono::steady_clock::now();
      float elapsed_seconds_init =
          std::chrono::duration_cast<std::chrono::milliseconds>(end_init -
                                                                begin_init)
              .count() /
          1000.;
      std::cout << "Model loaded in " << elapsed_seconds_init << " seconds\n";

      Wave wave = ReadWave(wave_filename);
      if (wave.samples.empty()) {
        std::cerr << "Failed to read: '" << wave_filename << "'\n";
        return -1;
      }

      std::cout << "Audio file: " << wave_filename << "\n";
      std::cout << "Sample rate: " << wave.sample_rate << " Hz\n";
      std::cout << "Duration: "
                << wave.samples.size() / static_cast<float>(wave.sample_rate)
                << " seconds\n";

      std::cout << "\nStart recognition...\n";
      const auto begin = std::chrono::steady_clock::now();

      OfflineStream stream = recognizer.CreateStream();
      stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                            wave.samples.size());

      recognizer.Decode(&stream);

      OfflineRecognizerResult result = recognizer.GetResult(&stream);

      const auto end = std::chrono::steady_clock::now();
      const float elapsed_seconds =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count() /
          1000.;
      float duration =
          wave.samples.size() / static_cast<float>(wave.sample_rate);
      float rtf = elapsed_seconds / duration;

      std::cout << "Text: " << result.text << "\n";
      std::cout << "Audio duration: " << duration << "s\n";
      std::cout << "Processing time: " << elapsed_seconds << "s\n";
      std::cout << "Real-time factor (RTF): " << rtf << "\n";

      return 0;
    }
  }

  std::cerr << "Error: Please provide an audio file.\n";
  std::cerr << kUsageMessage << "\n";
  return -1;
}

