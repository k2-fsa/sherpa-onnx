// sherpa-onnx/python/csrc/online-websocket-server.cc
//
// Copyright (c)  2025 Uniphore (Author: Manickavela A)

#include "sherpa-onnx/python/csrc/online-websocket-server-app.h"

#include <string>

#include "asio.hpp"
#include "sherpa-onnx/csrc/online-websocket-server.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void StartServerWrapper(py::list args) {
    int argc = args.size();
    std::vector<std::string> args_str;  // Store actual strings
    std::vector<char *> argv;           // Store pointers to those strings

    for (const auto &arg : args) {
        args_str.push_back(arg.cast<std::string>());
    }

    // Fill argv with pointers to the actual string data
    for (auto &str : args_str) {
        argv.push_back(str.data());
    }

    argv.push_back(nullptr);  // Null-terminate like C-style arrays

    // Call your server
    StartServer(argc, argv.data());
}

void PybindOnlineWebsocketServerWrapperApp(py::module *m) {
    m->def("start_server", &StartServerWrapper, "Start the WebSocket server",
    py::call_guard<py::gil_scoped_release>());
}

} // namespace sherpa_onnx