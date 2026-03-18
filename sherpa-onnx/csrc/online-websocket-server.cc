// sherpa-onnx/csrc/online-websocket-server.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation
// Copyright (c)  2025  Uniphore (Author: Manickavela A)

#include "sherpa-onnx/csrc/online-websocket-server.h"

int32_t main(int32_t argc, char *argv[]) {
  OnlineWebsocketServerApp app(argc, argv);
  app.Run();
  return 0;
}
