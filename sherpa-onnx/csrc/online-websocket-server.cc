// sherpa-onnx/csrc/online-websocket-server.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation
// Copyright (c)  2025  Uniphore (Author: Manickavela A)

#include <string>
#include <csignal>

#include "sherpa-onnx/csrc/online-websocket-server.h"

static constexpr const char *kUsageMessage = R"(
Automatic speech recognition with sherpa-onnx using websocket.

Usage:

./bin/sherpa-onnx-online-websocket-server --help

./bin/sherpa-onnx-online-websocket-server \
  --port=6006 \
  --num-work-threads=5 \
  --tokens=/path/to/tokens.txt \
  --encoder=/path/to/encoder.onnx \
  --decoder=/path/to/decoder.onnx \
  --joiner=/path/to/joiner.onnx \
  --log-file=./log.txt \
  --max-batch-size=5 \
  --loop-interval-ms=10

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)";


// Global server instance pointer for signal handling
OnlineWebsocketServerApp *global_server_instance = nullptr;

// Signal handler to stop the server
void SignalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        SHERPA_ONNX_LOGE("\nSignal %d received. Stopping server...", signal);
        if (global_server_instance) {
            global_server_instance->Stop();
        }
    }
}

OnlineWebsocketServerApp::OnlineWebsocketServerApp(
  int32_t argc, char *argv[]) : argc_(argc), argv_(argv) {}

void OnlineWebsocketServerApp::Run() {
  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OnlineWebsocketServerConfig config;

    // the server will listen on this port
    int32_t port = 6006;

    // size of the thread pool for handling network connections
    int32_t num_io_threads = 1;

    // size of the thread pool for neural network computation and decoding
    int32_t num_work_threads = 3;

    po.Register("num-io-threads", &num_io_threads,
                "Thread pool size for network connections.");

    po.Register("num-work-threads", &num_work_threads,
                "Thread pool size for for neural network "
                "computation and decoding.");

    po.Register("port", &port, "The port on which the server will listen.");

    config.Register(&po);

    if (argc_ == 1) {
        po.PrintUsage();
        exit(EXIT_FAILURE);
    }

    po.Read(argc_, argv_);

    if (po.NumArgs() != 0) {
        SHERPA_ONNX_LOGE("Unrecognized positional arguments!");
        po.PrintUsage();
        exit(EXIT_FAILURE);
    }

    config.Validate();

    // Set the global instance for signal handling
    global_server_instance = this;

    // Handle SIGINT and SIGTERM
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // io_conn for network connections
    // io_work for neural network and decoding

    sherpa_onnx::OnlineWebsocketServer server(io_conn_, io_work_, config);
    server.Run(port);

    SHERPA_ONNX_LOGE("Started!");
    SHERPA_ONNX_LOGE("Listening on: %d", port);
    SHERPA_ONNX_LOGE("Number of work threads: %d", num_work_threads);

    // give some work to do for the io_work pool
    auto work_guard = asio::make_work_guard(io_work_);

    std::vector<std::thread> io_threads;

    // decrement since the main thread is also used for network communications
    for (int32_t i = 0; i < num_io_threads - 1; ++i) {
        io_threads.emplace_back([this]() { io_conn_.run(); });
    }

    std::vector<std::thread> work_threads;
    for (int32_t i = 0; i < num_work_threads; ++i) {
        work_threads.emplace_back([this]() { io_work_.run(); });
    }

    // Main thread handles IO
    io_conn_.run();

    for (auto &t : io_threads) {
        t.join();
    }

    for (auto &t : work_threads) {
        t.join();
    }
    SHERPA_ONNX_LOGE("Server shut down gracefully.");
}

void OnlineWebsocketServerApp::Stop() {
    shutdown_requested_.store(true);
    io_conn_.stop();
    io_work_.stop();
    SHERPA_ONNX_LOGE("Server stopping...");
}

int32_t main(int32_t argc, char *argv[]) {
    OnlineWebsocketServerApp app(argc, argv);
    app.Run();
    return 0;
}

void StartServer(int32_t argc, char *argv[]) {
    OnlineWebsocketServerApp app(argc, argv);
    app.Run();
}