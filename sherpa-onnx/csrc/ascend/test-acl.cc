// sherpa-onnx/csrc/ascend/test-acl.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "acl/acl.h"

namespace sherpa_onnx {

static void TestAcl() { Acl acl; }

}  // namespace sherpa_onnx

int main() {
  sherpa_onnx::TestAcl();
  return 0;
}
