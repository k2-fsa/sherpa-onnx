// Copyright (c) 2026 LittleMouse
#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"

#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/axera/tts-session.h"

namespace sherpa_onnx {
TEST(AxEngineGuard, ContextSurvivesAnotherThreadsLastGuard) {
  const char *model = std::getenv("SHERPA_ONNX_AXERA_TEST_ENCODER");
  if (!model) GTEST_SKIP() << "Set SHERPA_ONNX_AXERA_TEST_ENCODER on AX650";
  std::promise<void> ready;
  std::promise<void> release;
  auto released = release.get_future();
  auto initialized = ready.get_future();
  std::thread worker([&] {
    AxEngineGuard guard;
    ready.set_value();
    released.wait();
  });
  initialized.wait();
  AxeraTtsSession session(model, false);
  release.set_value();
  worker.join();
  std::vector<int32_t> tokens(384, 0);
  EXPECT_NO_THROW({
    session.Set("cat_tokens", tokens.data(), tokens.size() * sizeof(int32_t));
    session.Run();
    EXPECT_EQ(session.Get("encoded").size(), 384 * 100);
  });
}
}  // namespace sherpa_onnx
