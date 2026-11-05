// sherpa-onnx/csrc/offline-cuda-iobinding-test.cc
//
// Copyright (c)  2026

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/offline-fire-red-asr-model.h"
#include "sherpa-onnx/csrc/offline-whisper-model.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {
namespace {

// These model-dependent tests are opt-in and never download assets. Set
// SHERPA_ONNX_RUN_CUDA_IO_TESTS=1 and the *_ENCODER / *_DECODER paths below.
// For a CPU-only ORT build, select --gtest_filter='CpuFallback.*'.
// For CUDA ORT, select --gtest_filter='CudaIoBinding.*:CudaAppendFallback.*'.
class ModelIoBindingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char *enabled = std::getenv("SHERPA_ONNX_RUN_CUDA_IO_TESTS");
    if (enabled == nullptr || std::string(enabled) != "1") {
      GTEST_SKIP() << "Set SHERPA_ONNX_RUN_CUDA_IO_TESTS=1 to run model tests";
    }
    const char *device = std::getenv("SHERPA_ONNX_CUDA_TEST_DEVICE");
    ASSERT_TRUE(GetCudaDeviceId({{"device_id", device ? device : "0"}},
                                &device_id_));
  }

  static bool HasCuda() {
    const auto providers = Ort::GetAvailableProviders();
    return std::find(providers.begin(), providers.end(),
                     "CUDAExecutionProvider") != providers.end();
  }

  int32_t device_id_ = 0;
};

class TempConfigFile {
 public:
  explicit TempConfigFile(const std::string &contents) {
#if defined(_WIN32)
    char temp_path[MAX_PATH];
    char temp_file[MAX_PATH];
    DWORD n = GetTempPathA(MAX_PATH, temp_path);
    if (n > 0 && n < MAX_PATH &&
        GetTempFileNameA(temp_path, "sio", 0, temp_file) != 0) {
      path_ = temp_file;
    }
#else
    char temp_path[] = "/tmp/sherpa_onnx_iobinding_test_XXXXXX";
    int fd = mkstemp(temp_path);
    if (fd != -1) {
      path_ = temp_path;
      if (close(fd) != 0) return;
    }
#endif
    if (path_.empty()) return;
    std::ofstream os(path_);
    os << contents;
    os.close();
    ok_ = static_cast<bool>(os);
  }

  ~TempConfigFile() {
    if (!path_.empty()) std::remove(path_.c_str());
  }
  TempConfigFile(const TempConfigFile &) = delete;
  TempConfigFile &operator=(const TempConfigFile &) = delete;
  bool Ok() const { return ok_; }
  std::string Provider() const { return "cuda:" + path_; }

 private:
  std::string path_;
  bool ok_ = false;
};

std::string RequiredModelPath(const char *variable) {
  const char *path = std::getenv(variable);
  if (path == nullptr || !std::ifstream(path, std::ios::binary).good()) {
    throw std::runtime_error(std::string("Set a readable model path in ") +
                             variable);
  }
  return path;
}

void CheckMemory(const Ort::Value &value, const char *name, int32_t device_id) {
  ASSERT_TRUE(value.IsTensor());
  auto info = value.GetTensorMemoryInfo();
  ASSERT_EQ(info.GetAllocatorName(), name);
  ASSERT_EQ(info.GetDeviceId(), device_id);
}

template <typename T>
Ort::Value Tensor(OrtAllocator *allocator, const std::vector<int64_t> &shape,
                  const std::vector<T> &data) {
  auto tensor =
      Ort::Value::CreateTensor<T>(allocator, shape.data(), shape.size());
  if (tensor.GetTensorTypeAndShapeInfo().GetElementCount() != data.size()) {
    throw std::runtime_error("Test tensor shape does not match data");
  }
  std::copy(data.begin(), data.end(),
            tensor.template GetTensorMutableData<T>());
  return tensor;
}

// The real model APIs return CPU logits/offset and reusable cross/self caches.
// Exercise several steps so a misplaced cache cannot hide behind one Run().
template <typename Model>
void CheckDecoder(Model *model, std::pair<Ort::Value, Ort::Value> cross,
                  std::vector<int64_t> tokens, const char *memory,
                  int32_t device_id) {
  ASSERT_NO_FATAL_FAILURE(CheckMemory(cross.first, memory, device_id));
  ASSERT_NO_FATAL_FAILURE(CheckMemory(cross.second, memory, device_id));
  auto self = model->GetInitialSelfKVCache();
  auto offset = Tensor<int64_t>(model->Allocator(), {1}, {0});
  for (int32_t step = 0; step != 3; ++step) {
    SCOPED_TRACE(step);
    const int64_t token_count = tokens.size();
    ASSERT_GT(token_count, 0);
    auto token_tensor =
        Tensor<int64_t>(model->Allocator(), {1, token_count}, tokens);
    auto out = model->ForwardDecoder(
        std::move(token_tensor), std::move(self.first), std::move(self.second),
        std::move(cross.first), std::move(cross.second), std::move(offset));
    ASSERT_NO_FATAL_FAILURE(CheckMemory(std::get<0>(out), "Cpu", 0));
    ASSERT_NO_FATAL_FAILURE(CheckMemory(std::get<1>(out), memory, device_id));
    ASSERT_NO_FATAL_FAILURE(CheckMemory(std::get<2>(out), memory, device_id));
    ASSERT_NO_FATAL_FAILURE(CheckMemory(std::get<3>(out), memory, device_id));
    ASSERT_NO_FATAL_FAILURE(CheckMemory(std::get<4>(out), memory, device_id));
    ASSERT_NO_FATAL_FAILURE(CheckMemory(std::get<5>(out), "Cpu", 0));
    if constexpr (std::tuple_size<decltype(out)>::value == 7) {
      if (std::get<6>(out)) {
        ASSERT_NO_FATAL_FAILURE(CheckMemory(std::get<6>(out), "Cpu", 0));
      }
    }
    const auto &logits = std::get<0>(out);
    auto info = logits.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    auto shape = info.GetShape();
    ASSERT_EQ(shape.size(), 3);
    ASSERT_GT(shape[2], 0);
    ASSERT_GE(info.GetElementCount(), static_cast<size_t>(shape[2]));
    const float *last = logits.template GetTensorData<float>() +
                        info.GetElementCount() - shape[2];
    tokens = {std::max_element(last, last + shape[2]) - last};
    self = {std::move(std::get<1>(out)), std::move(std::get<2>(out))};
    cross = {std::move(std::get<3>(out)), std::move(std::get<4>(out))};
    offset = std::move(std::get<5>(out));
    *offset.template GetTensorMutableData<int64_t>() += token_count;
  }
}

void CheckWhisper(const std::string &provider, const char *memory,
                  int32_t device_id, bool language_id = false) {
  OfflineModelConfig config;
  config.num_threads = 1;
  config.provider = provider;
  config.whisper.encoder =
      RequiredModelPath("SHERPA_ONNX_CUDA_TEST_WHISPER_ENCODER");
  config.whisper.decoder =
      RequiredModelPath("SHERPA_ONNX_CUDA_TEST_WHISPER_DECODER");
  std::unique_ptr<OfflineWhisperModel> model;
  if (language_id) {
    SpokenLanguageIdentificationConfig lid_config;
    lid_config.num_threads = 1;
    lid_config.provider = provider;
    lid_config.whisper.encoder = config.whisper.encoder;
    lid_config.whisper.decoder = config.whisper.decoder;
    model = std::make_unique<OfflineWhisperModel>(lid_config);
    ASSERT_TRUE(model->IsMultiLingual());
  } else {
    model = std::make_unique<OfflineWhisperModel>(config);
  }
  ASSERT_GT(model->FeatureDim(), 0);
  ASSERT_LE(model->FeatureDim(), 128);
  auto features = Tensor<float>(
      model->Allocator(), {1, model->FeatureDim(), 3000},
      std::vector<float>(model->FeatureDim() * 3000));
  auto cross = model->ForwardEncoder(std::move(features));
  if (language_id) {
    ASSERT_NO_FATAL_FAILURE(CheckMemory(cross.first, memory, device_id));
    ASSERT_NO_FATAL_FAILURE(CheckMemory(cross.second, memory, device_id));
    int32_t language = model->DetectLanguage(cross.first, cross.second, {});
    EXPECT_EQ(model->GetID2Lang().count(language), 1);
  } else {
    ASSERT_NO_FATAL_FAILURE(CheckDecoder(model.get(), std::move(cross),
                                         model->GetInitialTokens(), memory,
                                         device_id));
  }
}

void CheckFireRed(const std::string &provider, const char *memory,
                  int32_t device_id) {
  OfflineModelConfig config;
  config.num_threads = 1;
  config.provider = provider;
  config.fire_red_asr.encoder =
      RequiredModelPath("SHERPA_ONNX_CUDA_TEST_FIRERED_ENCODER");
  config.fire_red_asr.decoder =
      RequiredModelPath("SHERPA_ONNX_CUDA_TEST_FIRERED_DECODER");
  OfflineFireRedAsrModel model(config);
  auto features = Tensor<float>(model.Allocator(), {1, 100, 80},
                                std::vector<float>(100 * 80));
  auto lengths = Tensor<int64_t>(model.Allocator(), {1}, {100});
  auto cross =
      model.ForwardEncoder(std::move(features), std::move(lengths));
  ASSERT_NO_FATAL_FAILURE(CheckDecoder(
      &model, std::move(cross), {model.GetModelMetadata().sos_id}, memory,
      device_id));
}

using CudaIoBinding = ModelIoBindingTest;
using CpuFallback = ModelIoBindingTest;
using CudaAppendFallback = ModelIoBindingTest;

TEST_F(CudaIoBinding, WhisperConfigKeepsCudaCaches) {
  ASSERT_TRUE(HasCuda());
  TempConfigFile config("device_id=" + std::to_string(device_id_) + "\n");
  ASSERT_TRUE(config.Ok());
  ASSERT_NO_FATAL_FAILURE(CheckWhisper("cuda", "Cuda", 0));
  ASSERT_NO_FATAL_FAILURE(CheckWhisper(config.Provider(), "Cuda", device_id_));
}

TEST_F(CudaIoBinding, FireRedConfigKeepsCudaCaches) {
  ASSERT_TRUE(HasCuda());
  TempConfigFile config("device_id=" + std::to_string(device_id_) + "\n");
  ASSERT_TRUE(config.Ok());
  ASSERT_NO_FATAL_FAILURE(CheckFireRed("cuda", "Cuda", 0));
  ASSERT_NO_FATAL_FAILURE(CheckFireRed(config.Provider(), "Cuda", device_id_));
}

TEST_F(CudaIoBinding, WhisperLidUsesRequestedCudaConfig) {
  ASSERT_TRUE(HasCuda());
  TempConfigFile config("device_id=" + std::to_string(device_id_) + "\n");
  ASSERT_TRUE(config.Ok());
  ASSERT_NO_FATAL_FAILURE(CheckWhisper("cuda", "Cuda", 0, true));
  ASSERT_NO_FATAL_FAILURE(
      CheckWhisper(config.Provider(), "Cuda", device_id_, true));
}

TEST_F(CpuFallback, WhisperCachesStayOnCpu) {
  if (HasCuda()) GTEST_SKIP() << "Requires ONNX Runtime without CUDA EP";
  TempConfigFile config("device_id=1\n");
  ASSERT_TRUE(config.Ok());
  for (const auto &provider :
       {std::string("cpu"), std::string("cuda"), config.Provider()}) {
    SCOPED_TRACE(provider);
    ASSERT_NO_FATAL_FAILURE(CheckWhisper(provider, "Cpu", 0));
    ASSERT_NO_FATAL_FAILURE(CheckWhisper(provider, "Cpu", 0, true));
  }
}

TEST_F(CpuFallback, FireRedCachesStayOnCpu) {
  if (HasCuda()) GTEST_SKIP() << "Requires ONNX Runtime without CUDA EP";
  TempConfigFile config("device_id=1\n");
  ASSERT_TRUE(config.Ok());
  for (const auto &provider :
       {std::string("cpu"), std::string("cuda"), config.Provider()}) {
    SCOPED_TRACE(provider);
    ASSERT_NO_FATAL_FAILURE(CheckFireRed(provider, "Cpu", 0));
  }
}

TEST_F(CudaAppendFallback, WhisperCachesStayOnCpu) {
#if ORT_API_VERSION >= 12
  ASSERT_TRUE(HasCuda());
  TempConfigFile config("sherpa_test_invalid_cuda_option=1\n");
  ASSERT_TRUE(config.Ok());
  ASSERT_NO_FATAL_FAILURE(CheckWhisper(config.Provider(), "Cpu", 0));
#else
  GTEST_SKIP() << "Requires CUDA V2 provider options";
#endif
}

TEST_F(CudaAppendFallback, FireRedCachesStayOnCpu) {
#if ORT_API_VERSION >= 12
  ASSERT_TRUE(HasCuda());
  TempConfigFile config("sherpa_test_invalid_cuda_option=1\n");
  ASSERT_TRUE(config.Ok());
  ASSERT_NO_FATAL_FAILURE(CheckFireRed(config.Provider(), "Cpu", 0));
#else
  GTEST_SKIP() << "Requires CUDA V2 provider options";
#endif
}

}  // namespace
}  // namespace sherpa_onnx
