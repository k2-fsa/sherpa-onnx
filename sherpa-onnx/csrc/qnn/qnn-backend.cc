// sherpa-onnx/csrc/qnn/qnn-backend.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/qnn-backend.h"

#include <dlfcn.h>
#include <stdio.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "sherpa-onnx/csrc/qnn/macros.h"
#include "sherpa-onnx/csrc/qnn/utils.h"

namespace sherpa_onnx {

class QnnBackend::Impl {
 public:
  explicit Impl(const std::string &backend_lib) {
    bool ok = InitQnnInterface(backend_lib);
    if (!ok) {
      SHERPA_ONNX_LOGE("Failed to init qnn interface from '%s'",
                       backend_lib.c_str());
      return;
    }

    InitLog();
    InitBackend();
    InitDevice();

    is_initialized_ = true;
  }

  ~Impl() {
    if (context_handle_) {
      auto ret = qnn_interface_.contextFree(context_handle_, nullptr);
      SHERPA_ONNX_QNN_CHECK(ret, "Failed to call contextFree");
    }

    if (device_handle_) {
      auto ret = qnn_interface_.deviceFree(device_handle_);
      SHERPA_ONNX_QNN_CHECK(ret, "Failed to call deviceFree");
    }

    if (backend_handle_) {
      auto ret = qnn_interface_.backendFree(backend_handle_);
      SHERPA_ONNX_QNN_CHECK(ret, "Failed to call backendFree");
    }

    if (log_handle_) {
      auto ret = qnn_interface_.logFree(log_handle_);
      SHERPA_ONNX_QNN_CHECK(ret, "Failed to call logFree");
    }
  }

  void InitContext() {
    if (context_handle_) {
      SHERPA_ONNX_LOGE("context handle is already initialized");
      return;
    }

    auto ret = qnn_interface_.contextCreate(backend_handle_, device_handle_,
                                            context_config_, &context_handle_);
    SHERPA_ONNX_QNN_CHECK(ret, "Failed to call contextCreate");
  }

  void InitContext(Qnn_ContextHandle_t t) { context_handle_ = t; }

  Qnn_LogHandle_t LogHandle() const { return log_handle_; }

  Qnn_BackendHandle_t BackendHandle() const { return backend_handle_; }

  Qnn_DeviceHandle_t DeviceHandle() const { return device_handle_; }

  Qnn_ContextHandle_t ContextHandle() const { return context_handle_; }

  QNN_INTERFACE_VER_TYPE QnnInterface() const { return qnn_interface_; }

  QnnLog_Level_t LogLevel() const { return log_level_; }

  bool IsInitialized() const { return is_initialized_; }

 private:
  bool InitQnnInterface(const std::string &backend_lib) {
    backend_lib_handle_ = std::unique_ptr<void, decltype(&dlclose)>(
        dlopen(backend_lib.c_str(), RTLD_NOW | RTLD_LOCAL), &dlclose);
    if (!backend_lib_handle_) {
      SHERPA_ONNX_LOGE("Failed to dlopen '%s'. Error is: '%s'",
                       backend_lib.c_str(), dlerror());
      return false;
    }
    SHERPA_ONNX_LOGE("loaded %s", backend_lib.c_str());

    const char *symbol = "QnnInterface_getProviders";
    auto get_interface_providers =
        reinterpret_cast<QnnInterfaceGetProvidersFnType>(
            dlsym(backend_lib_handle_.get(), symbol));
    if (!get_interface_providers) {
      SHERPA_ONNX_LOGE("Failed to dlsym for '%s'. Error is: '%s'", symbol,
                       dlerror());
      return false;
    }
    SHERPA_ONNX_LOGE("Got %s", symbol);

    const QnnInterface_t **interface_providers = nullptr;
    uint32_t num_providers = 0;

    auto ret = get_interface_providers(&interface_providers, &num_providers);
    SHERPA_ONNX_QNN_CHECK(ret, "Failed to call get_interface_providers");

    if (!interface_providers) {
      SHERPA_ONNX_LOGE("interface_providers is nullptr");
      return false;
    }

    if (num_providers == 0) {
      SHERPA_ONNX_LOGE("Number of providers is 0");
      return false;
    }

    bool found_valid_interface = false;

    if (debug_) {
      SHERPA_ONNX_LOGE("QNN_API_VERSION_MAJOR: %d", QNN_API_VERSION_MAJOR);
      SHERPA_ONNX_LOGE("QNN_API_VERSION_MINOR: %d", QNN_API_VERSION_MINOR);
      SHERPA_ONNX_LOGE("QNN_API_VERSION_PATCH: %d", QNN_API_VERSION_PATCH);
    }

    for (size_t idx = 0; idx < num_providers; ++idx) {
      auto p = interface_providers[idx];

      if (debug_) {
        std::ostringstream os;
        os << "---" << idx << "----\n";
        os << "backendId: " << p->backendId << "\n";
        os << "coreApiVersion.major: " << p->apiVersion.coreApiVersion.major
           << "\n";
        os << "coreApiVersion.minor: " << p->apiVersion.coreApiVersion.minor
           << "\n";
        os << "coreApiVersion.patch: " << p->apiVersion.coreApiVersion.patch
           << "\n";

        os << "backendApiVersion.major: "
           << p->apiVersion.backendApiVersion.major << "\n";
        os << "backendApiVersion.minor: "
           << p->apiVersion.backendApiVersion.minor << "\n";
        os << "backendApiVersion.patch: "
           << p->apiVersion.backendApiVersion.patch << "\n";
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
      }

      qnn_interface_ = p->QNN_INTERFACE_VER_NAME;
      found_valid_interface = true;
      break;
    }

    if (!found_valid_interface) {
      SHERPA_ONNX_LOGE("Failed to find valid interface");
      return false;
    }

    if (debug_) {
      const char *build_id = nullptr;
      ret = qnn_interface_.backendGetBuildId(&build_id);
      SHERPA_ONNX_QNN_CHECK(ret, "Failed to call backendGetBuildId()");

      SHERPA_ONNX_LOGE("backend build ID: %s", build_id);
    }

    return true;
  }

  void InitLog() {
    auto ret = qnn_interface_.logCreate(LogCallback, log_level_, &log_handle_);
    SHERPA_ONNX_QNN_CHECK(ret, "Failed to call logCreate");
  }

  void InitBackend() {
    auto ret = qnn_interface_.backendCreate(log_handle_, backend_config_,
                                            &backend_handle_);
    SHERPA_ONNX_QNN_CHECK(ret, "Failed to call backendCreate");
  }

  void InitDevice() {
    auto ret =
        qnn_interface_.deviceCreate(log_handle_, nullptr, &device_handle_);
    SHERPA_ONNX_QNN_CHECK(ret, "Failed to call deviceCreate");
  }

 private:
  bool debug_ = true;
  std::unique_ptr<void, decltype(&dlclose)> backend_lib_handle_{nullptr,
                                                                &dlclose};

  QNN_INTERFACE_VER_TYPE qnn_interface_;

  QnnLog_Level_t log_level_ = QNN_LOG_LEVEL_WARN;
  // QnnLog_Level_t log_level_ = QNN_LOG_LEVEL_INFO;
  // QnnLog_Level_t log_level_ = QNN_LOG_LEVEL_VERBOSE;

  Qnn_LogHandle_t log_handle_ = nullptr;

  const QnnBackend_Config_t **backend_config_ = nullptr;
  Qnn_BackendHandle_t backend_handle_ = nullptr;

  Qnn_DeviceHandle_t device_handle_ = nullptr;

  Qnn_ContextHandle_t context_handle_ = nullptr;
  const QnnContext_Config_t **context_config_ = nullptr;
  bool is_initialized_ = false;
};

QnnBackend::~QnnBackend() = default;

QnnBackend::QnnBackend(const std::string &backend_lib)
    : impl_(std::make_unique<Impl>(backend_lib)) {}

void QnnBackend::InitContext() const { impl_->InitContext(); }

void QnnBackend::InitContext(Qnn_ContextHandle_t context_handle) const {
  impl_->InitContext(context_handle);
}

Qnn_LogHandle_t QnnBackend::LogHandle() const { return impl_->LogHandle(); }

Qnn_BackendHandle_t QnnBackend::BackendHandle() const {
  return impl_->BackendHandle();
}

Qnn_DeviceHandle_t QnnBackend::DeviceHandle() const {
  return impl_->DeviceHandle();
}

Qnn_ContextHandle_t QnnBackend::ContextHandle() const {
  return impl_->ContextHandle();
}

QNN_INTERFACE_VER_TYPE QnnBackend::QnnInterface() const {
  return impl_->QnnInterface();
}

QnnLog_Level_t QnnBackend::LogLevel() const { return impl_->LogLevel(); }

bool QnnBackend::IsInitialized() const { return impl_->IsInitialized(); }

}  // namespace sherpa_onnx
