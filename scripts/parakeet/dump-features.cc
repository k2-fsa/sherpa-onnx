// scripts/parakeet/dump-features.cc
//
// Copyright (c) 2026 Code Myriad

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/resample.h"

// Validation utility: float32 mono 16 kHz input, row-major float32 features.
// auto additionally exercises the recognizer's model-metadata selection.
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--version") {
      std::cout << OrtGetApiBase()->GetVersionString() << '\n';
      return 0;
    }
    if (argc < 4 || argc > 5) {
      throw std::runtime_error(
          "usage: parakeet-feature-dump legacy|reference|auto|resample "
          "input.f32 output.f32 [model-dir|input-sample-rate]");
    }
    const std::string mode = argv[1];
    if (mode != "legacy" && mode != "reference" && mode != "auto" &&
        mode != "resample") {
      throw std::runtime_error("unknown frontend mode");
    }
    std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open input");
    const auto size = input.tellg();
    if (size < 0 || size % sizeof(float)) {
      throw std::runtime_error("input must contain float32 samples");
    }
    std::vector<float> samples(static_cast<size_t>(size) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(samples.data()), size);
    if (!input && size != 0) throw std::runtime_error("incomplete input read");
    if (mode == "resample") {
      if (argc != 5) throw std::runtime_error("resample requires sample rate");
      const int32_t rate = std::stoi(argv[4]);
      if (rate <= 0) throw std::runtime_error("sample rate must be positive");
      sherpa_onnx::LinearResample resampler(
          rate, 16000, 0.99f * 0.5f * std::min(rate, 16000), 6);
      std::vector<float> output_samples;
      resampler.Resample(samples.data(), samples.size(), true, &output_samples);
      std::ofstream output(argv[3], std::ios::binary);
      output.write(reinterpret_cast<const char *>(output_samples.data()),
                   output_samples.size() * sizeof(float));
      if (!output) throw std::runtime_error("cannot write resampled samples");
      return 0;
    }

    sherpa_onnx::FeatureExtractorConfig config;
    config.feature_dim = 128;
    config.nemo_normalize_type = "per_feature";
    config.is_librosa = true;
    config.low_freq = 0;
    config.remove_dc_offset = false;
    config.parakeet_reference_frontend = mode == "reference";
    std::unique_ptr<sherpa_onnx::OfflineRecognizer> recognizer;
    std::unique_ptr<sherpa_onnx::OfflineStream> stream;
    if (mode == "auto") {
      if (argc != 5) throw std::runtime_error("auto requires model-dir");
      const std::string dir = argv[4];
      sherpa_onnx::OfflineRecognizerConfig rc;
      rc.model_config.transducer.encoder_filename = dir + "/encoder.onnx";
      rc.model_config.transducer.decoder_filename = dir + "/decoder.onnx";
      rc.model_config.transducer.joiner_filename = dir + "/joiner.onnx";
      rc.model_config.tokens = dir + "/tokens.txt";
      rc.model_config.model_type = "nemo_transducer";
      rc.model_config.num_threads = 4;
      recognizer = std::make_unique<sherpa_onnx::OfflineRecognizer>(rc);
      stream = recognizer->CreateStream();
    } else {
      stream = std::make_unique<sherpa_onnx::OfflineStream>(config);
    }
    if (!samples.empty()) {
      stream->AcceptWaveform(16000, samples.data(), samples.size());
    }
    const auto features = stream->GetFrames();
    if (mode == "auto") {
      std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> short_streams;
      std::vector<sherpa_onnx::OfflineStream *> batch;
      for (int32_t n : {0, 1, 159, 160, 319}) {
        auto short_stream = recognizer->CreateStream();
        if (n > 0) {
          std::vector<float> silence(n, 0.0f);
          short_stream->AcceptWaveform(16000, silence.data(), n);
        }
        batch.push_back(short_stream.get());
        short_streams.push_back(std::move(short_stream));
      }
      recognizer->DecodeStreams(batch.data(), batch.size());
      recognizer->DecodeStream(stream.get());
      const auto expected = stream->GetResult();
      batch.insert(batch.begin() + 2, stream.get());
      recognizer->DecodeStreams(batch.data(), batch.size());
      if (stream->GetResult().text != expected.text ||
          stream->GetResult().tokens != expected.tokens) {
        throw std::runtime_error("mixed short-input batch changed speech");
      }
      for (const auto &short_stream : short_streams) {
        const auto &r = short_stream->GetResult();
        if (!r.text.empty() || !r.tokens.empty() || !r.timestamps.empty()) {
          throw std::runtime_error("short-input result should be empty");
        }
      }
    }
    std::ofstream output(argv[3], std::ios::binary);
    output.write(reinterpret_cast<const char *>(features.data()),
                 features.size() * sizeof(float));
    if (!output) throw std::runtime_error("cannot write output");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
