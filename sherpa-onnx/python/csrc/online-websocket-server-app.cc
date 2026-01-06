// sherpa-onnx/python/csrc/online-websocket-server-app.cc
//
// Copyright (c)  2025  Uniphore (Author: Manickavela A)

#include "sherpa-onnx/python/csrc/online-websocket-server-app.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/online-websocket-server.h"

namespace sherpa_onnx {

static void StartServerWrapper(py::list args) {
  int argc = static_cast<int>(args.size());
  std::vector<std::string> args_str;
  std::vector<char *> argv;

  args_str.reserve(argc);
  for (const auto &arg : args) {
    args_str.push_back(arg.cast<std::string>());
  }

  argv.reserve(argc + 1);
  for (auto &str : args_str) {
    argv.push_back(str.data());
  }
  argv.push_back(nullptr);

  StartServer(argc, argv.data());
}

void PybindOnlineWebsocketServerApp(py::module *m) {
  m->def("start_online_websocket_server", &StartServerWrapper,
         py::arg("args"),
         "Start the online websocket server with command line arguments");
}

}  // namespace sherpa_onnx
