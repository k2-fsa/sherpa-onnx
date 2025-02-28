// sherpa-onnx/csrc/online-wenet-ctc-model-config.h
//
// Copyright (c)  2025  Uniphore (Author: Manickavela A)s

#ifndef SHERPA_ONNX_ONLINE_WEBSOCKET_SERVER_H
#define SHERPA_ONNX_ONLINE_WEBSOCKET_SERVER_H

#include <asio.hpp>
#include <thread>
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-websocket-server-impl.h"
#include "sherpa-onnx/csrc/parse-options.h"

class OnlineWebsocketServerApp {
public:
    OnlineWebsocketServerApp(int32_t argc, char *argv[]);
    void Run();
    void Stop();

private:
    int32_t argc_;
    char **argv_;
    asio::io_context io_conn_;  // ASIO context for connections
    asio::io_context io_work_;  // ASIO context for work
    std::atomic<bool> shutdown_requested_{false};
    std::vector<std::thread> io_threads_;
    std::vector<std::thread> work_threads_;
};

// Declare StartServer so it's accessible for Pybind
void StartServer(int32_t argc, char *argv[]);

#endif // SHERPA_ONNX_ONLINE_WEBSOCKET_SERVER_H
