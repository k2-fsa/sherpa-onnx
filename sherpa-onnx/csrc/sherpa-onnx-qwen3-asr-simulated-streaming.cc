// sherpa-onnx/csrc/sherpa-onnx-qwen3-asr-simulated-streaming.cc
//
// Copyright (c) 2026 Qing Peng
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/qwen3-asr-simulated-streaming.h"

namespace {
using Status = sherpa_onnx::Qwen3ASRStreamingStatus;
using Clock = std::chrono::steady_clock;

const char *StatusName(Status status) {
  switch (status) {
    case Status::kOk:
      return "ok";
    case Status::kInvalidArgument:
      return "invalid_argument";
    case Status::kInputFinished:
      return "input_finished";
    case Status::kAudioLimit:
      return "audio_limit";
    case Status::kContextLimit:
      return "context_limit";
    case Status::kTokenLimit:
      return "token_limit";
    case Status::kRepetitionStopped:
      return "repetition_stopped";
    default:
      return "decode_error";
  }
}

uint32_t ReadLe(const unsigned char *p, int n) {
  uint32_t value = 0;
  for (int i = 0; i < n; ++i) value |= uint32_t{p[i]} << (8 * i);
  return value;
}

// Read the validated data chunk directly, using one bounded RIFF parser.
// This CLI accepts canonical PCM16 WAV and bounded metadata, at most 30 s.
Status ReadBoundedWave(const std::string &path, std::vector<float> *pcm) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) return Status::kInvalidArgument;
  const std::streamoff size = file.tellg();
  if (size < 44 || size > 2 * 480000 + 65536) return Status::kAudioLimit;
  file.seekg(0);
  std::array<unsigned char, 16> bytes{};
  file.read(reinterpret_cast<char *>(bytes.data()), 12);
  if (!file || ReadLe(bytes.data(), 4) != 0x46464952 ||
      ReadLe(bytes.data() + 8, 4) != 0x45564157 ||
      ReadLe(bytes.data() + 4, 4) > size - 8)
    return Status::kInvalidArgument;
  bool format_ok = false;
  while (file && file.tellg() <= size - 8) {
    file.read(reinterpret_cast<char *>(bytes.data()), 8);
    auto id = ReadLe(bytes.data(), 4);
    auto n = ReadLe(bytes.data() + 4, 4);
    if (n > size - static_cast<std::streamoff>(file.tellg())) {
      return Status::kInvalidArgument;
    }
    if (id == 0x20746d66) {
      if (n != 16) return Status::kInvalidArgument;
      file.read(reinterpret_cast<char *>(bytes.data()), 16);
      format_ok = ReadLe(bytes.data(), 2) == 1 &&
                  ReadLe(bytes.data() + 2, 2) == 1 &&
                  ReadLe(bytes.data() + 4, 4) == 16000 &&
                  ReadLe(bytes.data() + 12, 2) == 2 &&
                  ReadLe(bytes.data() + 14, 2) == 16;
    } else if (id == 0x61746164) {
      if (!format_ok || n % 2) return Status::kInvalidArgument;
      if (n / 2 > 480000) return Status::kAudioLimit;
      std::vector<unsigned char> data(n);
      if (n) file.read(reinterpret_cast<char *>(data.data()), n);
      if (!file) return Status::kInvalidArgument;
      pcm->resize(n / 2);
      for (size_t i = 0; i < pcm->size(); ++i) {
        int32_t value = ReadLe(data.data() + 2 * i, 2);
        if (value >= 32768) value -= 65536;
        (*pcm)[i] = value / 32768.0f;
      }
      return Status::kOk;
    } else {
      file.seekg(n, std::ios::cur);
    }
    if (n % 2) file.seekg(1, std::ios::cur);
  }
  return Status::kInvalidArgument;
}

void Emit(const sherpa_onnx::Qwen3ASRStreamingOutcome &out,
          const sherpa_onnx::Qwen3ASRStreamingUpdate &current, int file_index,
          bool trace, double seconds) {
  auto write = [&](const auto &update, Status status) {
    nlohmann::json row = {
        {"file_index", file_index},
        {"status", StatusName(status)},
        {"revision", update.revision},
        {"audio_end_sample", update.audio_end_sample},
        {"is_final", update.is_final},
        {"had_invalid_utf8", update.had_invalid_utf8},
        {"accepted_samples", out.accepted_samples},
        {"processed_samples", out.processed_samples},
        {"result", nlohmann::json::parse(update.result.AsJsonString())}};
    if (trace) {
      row["call_seconds"] = seconds;
      row["decode"] = nullptr;
      for (const auto &step : out.steps) {
        if (status != Status::kOk &&
            step.audio_end_sample <= update.audio_end_sample)
          continue;
        if (step.audio_end_sample != update.audio_end_sample &&
            status == Status::kOk)
          continue;
        row["decode"] = {{"audio_end_sample", step.audio_end_sample},
                         {"prompt_tokens", step.prompt_tokens},
                         {"retained_tokens", step.retained_tokens},
                         {"generated_tokens", step.generated_tokens},
                         {"context_limit", step.context_limit},
                         {"stop_reason", step.stop_reason},
                         {"service_seconds", step.service_seconds},
                         {"encoder_seconds", step.encoder_seconds},
                         {"decoder_seconds", step.decoder_seconds},
                         {"decoder_calls", step.decoder_calls}};
      }
    }
    std::cout << row.dump() << '\n';
  };
  for (const auto &u : out.updates) write(u, Status::kOk);
  if (out.status != Status::kOk) write(current, out.status);
}
}  // namespace

int main(int argc, char *argv[]) {
  sherpa_onnx::ParseOptions po(
      "Qwen3-ASR cumulative-audio file replay (16 kHz mono PCM16 WAV).\n"
      "Usage: sherpa-onnx-qwen3-asr-simulated-streaming [options] file.wav "
      "...\n"
      "Replay runs as fast as possible; output wall times are not arrival "
      "times.");
  sherpa_onnx::OfflineRecognizerConfig model;
  sherpa_onnx::Qwen3ASRSimulatedStreamingConfig config;
  sherpa_onnx::Qwen3ASRSimulatedStreamingOptions options;
  int32_t packet = 320;
  std::string mode = "rollback";
  bool trace = false;
  model.Register(&po);
  po.Register("chunk-size-samples", &config.chunk_size_samples,
              "Audio boundary");
  po.Register("unfixed-chunk-num", &config.unfixed_chunk_num, "Warmup rounds");
  po.Register("unfixed-token-num", &config.unfixed_token_num, "Token rollback");
  po.Register("max-utterance-samples", &config.max_utterance_samples,
              "Audio limit");
  po.Register("packet-size-samples", &packet, "Packet samples (default 20 ms)");
  po.Register("prefix-mode", &mode, "rollback or none");
  po.Register("language", &options.language,
              "Canonical language, e.g. Chinese");
  po.Register("trace", &trace, "Include service-time measurements");
  std::vector<const char *> args{argv[0], "--print-args=false"};
  for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
  po.Read(args.size(), args.data());
  if (!po.NumArgs()) {
    po.PrintUsage();
    return 1;
  }
  try {
    if (packet <= 0 || packet > 480000 ||
        (mode != "rollback" && mode != "none")) {
      throw std::invalid_argument("Invalid replay configuration");
    }
    if (mode == "none")
      config.unfixed_chunk_num = std::numeric_limits<int32_t>::max();
    // CLI hotwords are an immutable utterance option; an empty flag disables.
    options.hotwords = model.model_config.qwen3_asr.hotwords;
    auto start = Clock::now();
    sherpa_onnx::Qwen3ASRSimulatedStreamingRecognizer recognizer(model, config);
    double init = std::chrono::duration<double>(Clock::now() - start).count();
    std::cerr << nlohmann::json(
                     {{"prefix_mode", mode},
                      {"chunk_size_samples", config.chunk_size_samples},
                      {"unfixed_chunk_num", config.unfixed_chunk_num},
                      {"unfixed_token_num", config.unfixed_token_num},
                      {"packet_size_samples", packet},
                      {"initialization_seconds", init}})
                     .dump()
              << '\n';
    bool failed = false;
    for (int index = 1; index <= po.NumArgs(); ++index) {
      auto stream = recognizer.CreateStream(options);
      std::vector<float> pcm;
      auto status = ReadBoundedWave(po.GetArg(index), &pcm);
      if (status != Status::kOk) {
        sherpa_onnx::Qwen3ASRStreamingOutcome out;
        out.status = status;
        Emit(out, stream->GetResult(), index, trace, 0);
        failed = true;
        continue;
      }
      for (size_t offset = 0; offset < pcm.size() && status == Status::kOk;) {
        auto n = std::min<size_t>(packet, pcm.size() - offset);
        start = Clock::now();
        auto out = stream->AcceptWaveform(pcm.data() + offset, n);
        auto elapsed =
            std::chrono::duration<double>(Clock::now() - start).count();
        Emit(out, stream->GetResult(), index, trace, elapsed);
        status = out.status;
        offset += n;
      }
      if (status == Status::kOk) {
        start = Clock::now();
        auto out = stream->Finish();
        auto elapsed =
            std::chrono::duration<double>(Clock::now() - start).count();
        Emit(out, stream->GetResult(), index, trace, elapsed);
        status = out.status;
      }
      failed |= status != Status::kOk;
    }
    return failed ? 1 : 0;
  } catch (const std::exception &) {
    std::cerr
        << "Qwen3-ASR replay failed: check configuration and model files\n";
    return 1;
  }
}
