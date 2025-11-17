// sherpa-onnx/csrc/qnn/qnn-model.h
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/qnn-model.h"

#include <dlfcn.h>

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/qnn/macros.h"
#include "sherpa-onnx/csrc/qnn/qnn-backend.h"
#include "sherpa-onnx/csrc/qnn/utils.h"

namespace sherpa_onnx {

class QnnModel::Impl {
 public:
  Impl(const std::string &model_so, const QnnBackend *backend, bool debug)
      : debug_(debug), backend_(backend) {
    bool ok = InitModel(model_so);
    if (!ok) {
      SHERPA_ONNX_LOGE("Failed to load '%s'", model_so.c_str());
      return;
    }

    ok = InitSymbols();
    if (!ok) {
      SHERPA_ONNX_LOGE("Failed to get model symbols from '%s'",
                       model_so.c_str());
      return;
    }

    InitGraph();

    PostInit();
  }

  Impl(const std::string &binary_context_file, const std::string &system_lib,
       const QnnBackend *backend, BinaryContextTag, bool debug)
      : debug_(debug), backend_(backend) {
    bool ok = LoadSystemLib(binary_context_file, system_lib);
    if (!ok) {
      return;
    }

    PostInit();
  }

  bool LoadSystemLib(const std::string &binary_context_file,
                     const std::string &system_lib) {
    system_lib_handle_ = std::unique_ptr<void, decltype(&dlclose)>(
        dlopen(system_lib.c_str(), RTLD_NOW | RTLD_LOCAL), &dlclose);
    if (!system_lib_handle_) {
      SHERPA_ONNX_LOGE("Failed to dlopen '%s'. Error is: '%s'",
                       system_lib.c_str(), dlerror());
      return false;
    }
    if (debug_) {
      SHERPA_ONNX_LOGE("loaded %s", system_lib.c_str());
    }

    auto get_system_interface_providers =
        reinterpret_cast<QnnSystemInterfaceGetProvidersFnType>(
            dlsym(system_lib_handle_.get(), "QnnSystemInterface_getProviders"));

    if (!get_system_interface_providers) {
      SHERPA_ONNX_LOGE("Failed to get QnnSystemInterface_getProviders");
      return false;
    }

    const QnnSystemInterface_t **system_interface_providers = nullptr;
    uint32_t num_providers = 0;
    if (get_system_interface_providers(&system_interface_providers,
                                       &num_providers) != QNN_SUCCESS) {
      SHERPA_ONNX_LOGE("Failed to get system interface providers.");
      return false;
    }

    if (!system_interface_providers) {
      SHERPA_ONNX_LOGE(
          "Failed to get system interface providers: null "
          "interface providers received.");
      return false;
    }

    if (!num_providers) {
      SHERPA_ONNX_LOGE(
          "Failed to get interface providers: 0 interface providers.");
      return false;
    }

    for (uint32_t i = 0; i < num_providers; ++i) {
      if (debug_) {
        SHERPA_ONNX_LOGE("QNN_SYSTEM_API_VERSION_MAJOR: %d",
                         static_cast<int32_t>(QNN_SYSTEM_API_VERSION_MAJOR));
        SHERPA_ONNX_LOGE("QNN_SYSTEM_API_VERSION_MINOR: %d",
                         static_cast<int32_t>(QNN_SYSTEM_API_VERSION_MINOR));
        SHERPA_ONNX_LOGE(
            "systemApiVersion.major: %d",
            static_cast<int32_t>(
                system_interface_providers[i]->systemApiVersion.major));
        SHERPA_ONNX_LOGE(
            "systemApiVersion.minor: %d",
            static_cast<int32_t>(
                system_interface_providers[i]->systemApiVersion.minor));
      }

      qnn_system_interface_ =
          system_interface_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
    }

    // read file into a buffer
    std::vector<uint8_t> buffer = ReadFile<uint8_t>(binary_context_file);

    QnnSystemContext_Handle_t sys_ctx_handle = nullptr;
    if (qnn_system_interface_.systemContextCreate(&sys_ctx_handle) !=
        QNN_SUCCESS) {
      SHERPA_ONNX_LOGE("Could not create system handle.");
      return false;
    }

    const QnnSystemContext_BinaryInfo_t *binary_info = nullptr;
    Qnn_ContextBinarySize_t binary_info_size = 0;

    auto ret = qnn_system_interface_.systemContextGetBinaryInfo(
        sys_ctx_handle, static_cast<void *>(buffer.data()), buffer.size(),
        &binary_info, &binary_info_size);
    if (ret != QNN_SUCCESS) {
      SHERPA_ONNX_LOGE("Failed to get context binary info from '%s'",
                       binary_context_file.c_str());

      qnn_system_interface_.systemContextFree(sys_ctx_handle);
      return false;
    }

    const GraphConfigInfo **graph_configs_info = nullptr;

    uint32_t graph_configs_info_count = 0;
    GraphInfo **graphs_info = nullptr;
    uint32_t graphs_count = 0;

    if (!CopyMetadataToGraphsInfo(binary_info, graphs_info, graphs_count)) {
      SHERPA_ONNX_LOGE("Failed to call CopyMetadataToGraphsInfo");

      qnn_system_interface_.systemContextFree(sys_ctx_handle);
      return false;
    }

    qnn_system_interface_.systemContextFree(sys_ctx_handle);

    auto free_graphs_info = [&graphs_info, &graphs_count] {
      for (uint32_t i = 0; i < graphs_count; ++i) {
        for (uint32_t k = 0; k < graphs_info[i]->num_input_tensors; ++k) {
          FreeTensor(&graphs_info[i]->input_tensors[k]);
        }

        for (uint32_t k = 0; k < graphs_info[i]->num_output_tensors; ++k) {
          FreeTensor(&graphs_info[i]->output_tensors[k]);
        }

        free(graphs_info[i]->input_tensors);
        free(graphs_info[i]->output_tensors);

        free(graphs_info[i]->graph_name);
      }

      free(graphs_info[0]);
      free(graphs_info);
    };

    if (graphs_count > 1) {
      SHERPA_ONNX_LOGE("Only the first graph is used");
    }

    Qnn_ContextHandle_t context_handle = nullptr;

    if (backend_->QnnInterface().contextCreateFromBinary(
            backend_->BackendHandle(), backend_->DeviceHandle(),
            context_config_, static_cast<void *>(buffer.data()), buffer.size(),
            &context_handle, nullptr) != QNN_SUCCESS) {
      free_graphs_info();
      SHERPA_ONNX_LOGE("Could not create context from binary.");
      return false;
    }

    backend_->InitContext(context_handle);

    if (backend_->QnnInterface().graphRetrieve(
            context_handle, (*graphs_info)[0].graph_name,
            &((*graphs_info)[0].graph)) != QNN_SUCCESS) {
      free_graphs_info();
      SHERPA_ONNX_LOGE("Unable to retrieve graph handle for graph %d", 0);
      return false;
    }

    graph_handle_ = (*graphs_info)[0].graph;

    InitInputTensors((*graphs_info)[0]);
    InitOutputTensors((*graphs_info)[0]);

    free_graphs_info();

    return true;
  }

  ~Impl() = default;

  bool SaveBinaryContext(const std::string &filename) {
    auto qnn_interface = backend_->QnnInterface();

    if (!qnn_interface.contextGetBinarySize ||
        !qnn_interface.contextGetBinary) {
      SHERPA_ONNX_LOGE(
          "contextGetBinarySizeFnHandle or "
          "contextGetBinaryFnHandle is nullptr.");
      return false;
    }

    uint64_t required_buffer_size{0};
    auto ret = qnn_interface.contextGetBinarySize(backend_->ContextHandle(),
                                                  &required_buffer_size);
    SHERPA_ONNX_QNN_CHECK(ret, "Failed to call contextGetBinarySize");

    if (debug_) {
      SHERPA_ONNX_LOGE("context binary size: %.3f MB",
                       static_cast<float>(required_buffer_size) / 1024 / 1024);
    }
    std::vector<uint8_t> saveBuffer(required_buffer_size);
    uint64_t writtenBufferSize{0};

    ret = qnn_interface.contextGetBinary(
        backend_->ContextHandle(), reinterpret_cast<void *>(saveBuffer.data()),
        required_buffer_size, &writtenBufferSize);

    SHERPA_ONNX_QNN_CHECK(ret, "Failed to call contextGetBinary");

    if (required_buffer_size < writtenBufferSize) {
      SHERPA_ONNX_LOGE(
          "Illegal written buffer size %d bytes. Cannot exceed "
          "allocated memory of %d bytes",
          static_cast<int32_t>(writtenBufferSize),
          static_cast<int32_t>(required_buffer_size));
      return false;
    }
    std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
    if (!ofs) {
      SHERPA_ONNX_LOGE("Failed to create '%s'", filename.c_str());
      return false;
    }

    ofs.write(reinterpret_cast<const char *>(saveBuffer.data()),
              saveBuffer.size());

    if (!ofs) {
      SHERPA_ONNX_LOGE("Failed to write '%s'", filename.c_str());
      return false;
    }

    return true;
  }

  const std::vector<std::string> &InputTensorNames() const {
    return input_tensor_names_;
  }

  const std::vector<std::string> &OutputTensorNames() const {
    return output_tensor_names_;
  }

  std::vector<int32_t> TensorShape(const std::string &name) const {
    std::vector<int32_t> shape;

    if (!HasTensor(name)) {
      SHERPA_ONNX_LOGE("No such tensor '%s'", name.c_str());
      return shape;
    }

    auto t = name2tensor_.at(name);

    shape = {t->v1.dimensions, t->v1.dimensions + t->v1.rank};

    return shape;
  }

  int32_t TensorSizeInBytes(const std::string &name) const {
    if (!HasTensor(name)) {
      return 0;
    }

    return name2tensor_.at(name)->v1.clientBuf.dataSize;
  }

  bool HasTensor(const std::string &name) const {
    return name2tensor_.count(name);
  }

  bool SetInputTensorData(const std::string &name, const float *p, int32_t n) {
    if (!HasTensor(name)) {
      SHERPA_ONNX_LOGE("No such tensor '%s'", name.c_str());
      return false;
    }

    auto t = name2tensor_.at(name);
    if (t->v1.dataType != QNN_DATATYPE_UFIXED_POINT_16) {
      SHERPA_ONNX_LOGE(
          "tensor '%s' should be of type "
          "QNN_DATATYPE_UFIXED_POINT_16, but it is %s",
          name.c_str(), TensorDataTypeToString(t->v1.dataType).c_str());
      return false;
    }

    if (t->v1.quantizeParams.quantizationEncoding !=
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      SHERPA_ONNX_LOGE(
          "tensor '%s' should be quantized with "
          "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET, but it is %s",
          name.c_str(),
          QuantizationEncodingToString(
              t->v1.quantizeParams.quantizationEncoding)
              .c_str());
      return false;
    }

    if (n * sizeof(uint16_t) != t->v1.clientBuf.dataSize) {
      SHERPA_ONNX_LOGE("tensor '%s' expects %d bytes, but you provide %d bytes",
                       name.c_str(),
                       static_cast<int32_t>(t->v1.clientBuf.dataSize),
                       static_cast<int32_t>(n * sizeof(uint16_t)));
      return false;
    }

    FillData(t, p, n);

    return true;
  }

  bool SetInputTensorData(const std::string &name, const int32_t *p,
                          int32_t n) {
    if (!HasTensor(name)) {
      SHERPA_ONNX_LOGE("No such tensor '%s'", name.c_str());
      return false;
    }

    auto t = name2tensor_.at(name);
    if (t->v1.dataType != QNN_DATATYPE_INT_32) {
      SHERPA_ONNX_LOGE(
          "tensor '%s' should be of type "
          "QNN_DATATYPE_INT_32, but it is %s",
          name.c_str(), TensorDataTypeToString(t->v1.dataType).c_str());
      return false;
    }

    if (n * sizeof(int32_t) != t->v1.clientBuf.dataSize) {
      SHERPA_ONNX_LOGE("tensor '%s' expects %d bytes, but you provide %d bytes",
                       name.c_str(),
                       static_cast<int32_t>(t->v1.clientBuf.dataSize),
                       static_cast<int32_t>(n * sizeof(int32_t)));
      return false;
    }

    FillData(t, p, n);

    return true;
  }

  std::vector<float> GetOutputTensorData(const std::string &name) {
    if (!HasTensor(name)) {
      SHERPA_ONNX_LOGE("No such tensor '%s'", name.c_str());
      return {};
    }

    auto t = name2tensor_.at(name);
    if (t->v1.dataType != QNN_DATATYPE_UFIXED_POINT_16) {
      SHERPA_ONNX_LOGE(
          "tensor '%s' should be of type "
          "QNN_DATATYPE_UFIXED_POINT_16, but it is %s",
          name.c_str(), TensorDataTypeToString(t->v1.dataType).c_str());
      return {};
    }

    if (t->v1.quantizeParams.quantizationEncoding !=
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      SHERPA_ONNX_LOGE(
          "tensor '%s' should be quantized with "
          "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET, but it is %s",
          name.c_str(),
          QuantizationEncodingToString(
              t->v1.quantizeParams.quantizationEncoding)
              .c_str());
      return {};
    }

    int32_t n = t->v1.clientBuf.dataSize / sizeof(uint16_t);
    std::vector<float> ans(n);

    GetData(t, ans.data(), n);

    return ans;
  }

  bool Run() {
    std::vector<Qnn_Tensor_t> input_tensors_raw;
    std::vector<Qnn_Tensor_t> output_tensors_raw;

    input_tensors_raw.reserve(input_tensors_.size());
    output_tensors_raw.reserve(output_tensors_.size());

    for (const auto &p : input_tensors_) {
      input_tensors_raw.push_back(*p);
    }

    for (const auto &p : output_tensors_) {
      output_tensors_raw.push_back(*p);
    }

    auto ret = backend_->QnnInterface().graphExecute(
        graph_handle_, input_tensors_raw.data(), input_tensors_raw.size(),
        output_tensors_raw.data(), output_tensors_raw.size(), nullptr, nullptr);
    SHERPA_ONNX_QNN_CHECK(ret, "Failed to run graphExecute");

    return true;
  }

  bool IsInitialized() const { return is_initialized_; }

 private:
  void PostInit() {
    AllocateBuffer();
    SetupPointers();

    is_initialized_ = true;
  }

  bool InitModel(const std::string &model_so) {
    model_lib_handle_ = std::unique_ptr<void, decltype(&dlclose)>(
        dlopen(model_so.c_str(), RTLD_NOW | RTLD_LOCAL), &dlclose);
    if (!model_lib_handle_) {
      SHERPA_ONNX_LOGE("Failed to dlopen '%s'. Error is: '%s'",
                       model_so.c_str(), dlerror());
      return false;
    }

    if (debug_) {
      SHERPA_ONNX_LOGE("loaded %s", model_so.c_str());
    }

    return true;
  }

  bool InitSymbols() {
    const char *symbol = "QnnModel_composeGraphs";

    compose_graphs_fn_handle_ = reinterpret_cast<ComposeGraphsFnHandleType>(
        dlsym(model_lib_handle_.get(), symbol));
    if (!compose_graphs_fn_handle_) {
      SHERPA_ONNX_LOGE("Failed to dlsym for '%s'. Error is: '%s'", symbol,
                       dlerror());
      return false;
    }

    symbol = "QnnModel_freeGraphsInfo";
    free_graph_info_fn_handle_ = reinterpret_cast<FreeGraphInfoFnHandleType>(
        dlsym(model_lib_handle_.get(), symbol));
    if (!free_graph_info_fn_handle_) {
      SHERPA_ONNX_LOGE("Failed to dlsym for '%s'. Error is: '%s'", symbol,
                       dlerror());
      return false;
    }
    return true;
  }

  void InitGraph() {
    const GraphConfigInfo **graph_configs_info = nullptr;

    uint32_t graph_configs_info_count = 0;
    GraphInfo **graphs_info = nullptr;
    uint32_t graphs_count = 0;

    auto ret = compose_graphs_fn_handle_(
        backend_->BackendHandle(), backend_->QnnInterface(),
        backend_->ContextHandle(), graph_configs_info, graph_configs_info_count,
        &graphs_info, &graphs_count, debug_, LogCallback, backend_->LogLevel());
    SHERPA_ONNX_QNN_CHECK(ret, "Failed to call compose_graphs_fn_handle_");

    if (debug_) {
      SHERPA_ONNX_LOGE("graphs_count: %d", (int32_t)graphs_count);
    }

    for (uint32_t i = 0; i < graphs_count; ++i) {
      if (debug_) {
        SHERPA_ONNX_LOGE(
            "Finalizing graph %d/%d: '%s'", static_cast<int32_t>(i),
            static_cast<int32_t>(graphs_count), (*graphs_info)[i].graph_name);
      }
      ret = backend_->QnnInterface().graphFinalize((*graphs_info)[i].graph,
                                                   nullptr, nullptr);
      SHERPA_ONNX_QNN_CHECK(ret, "Failed to call graph_finalize");
    }

    if (graphs_count > 1) {
      SHERPA_ONNX_LOGE("We only use the first graph: %s",
                       (*graphs_info)[0].graph_name);
    }

    InitInputTensors((*graphs_info)[0]);
    InitOutputTensors((*graphs_info)[0]);

    graph_handle_ = (*graphs_info)[0].graph;
  }

  void InitInputTensors(GraphInfo graph) {
    input_tensors_.reserve(graph.num_input_tensors);
    input_tensor_names_.reserve(graph.num_input_tensors);

    for (uint32_t i = 0; i < graph.num_input_tensors; ++i) {
      auto p = TensorPtr(new Qnn_Tensor_t(QNN_TENSOR_INIT), &FreeTensor);

      CopyTensorInfo(graph.input_tensors[i], *p);

      if (debug_) {
        SHERPA_ONNX_LOGE("input %d", (int)i);
        PrintTensor(p->v2);
      }

      std::string name = p->v1.name;
      name2tensor_[name] = p.get();
      input_tensor_names_.push_back(std::move(name));

      input_tensors_.push_back(std::move(p));
    }
  }

  void InitOutputTensors(GraphInfo graph) {
    output_tensors_.reserve(graph.num_output_tensors);
    output_tensor_names_.reserve(graph.num_output_tensors);
    for (uint32_t i = 0; i < graph.num_output_tensors; ++i) {
      auto p = TensorPtr(new Qnn_Tensor_t(QNN_TENSOR_INIT), &FreeTensor);

      CopyTensorInfo(graph.output_tensors[i], *p);

      if (debug_ && (i + 3 > graph.num_output_tensors)) {
        SHERPA_ONNX_LOGE("output %d", (int)i);

        PrintTensor(p->v2);
      }

      std::string name = p->v1.name;
      name2tensor_[name] = p.get();
      output_tensor_names_.push_back(std::move(name));

      output_tensors_.push_back(std::move(p));
    }
  }

  void AllocateBuffer() {
    uint32_t n = 0;
    for (const auto &p : name2tensor_) {
      n += p.second->v1.clientBuf.dataSize;
    }

    if (debug_) {
      SHERPA_ONNX_LOGE("Allocate %d bytes, or %.3f MB", static_cast<int32_t>(n),
                       static_cast<float>(n) / 1024 / 1024);
    }

    buffer_.resize(n);
  }

  void SetupPointers() {
    uint8_t *p = buffer_.data();
    uint32_t n = 0;
    for (auto &t : input_tensors_) {
      t->v1.clientBuf.data = p;
      p += t->v1.clientBuf.dataSize;
    }

    for (auto &t : output_tensors_) {
      t->v1.clientBuf.data = p;
      p += t->v1.clientBuf.dataSize;
    }

    if (debug_) {
      if (p == buffer_.data() + buffer_.size()) {
        SHERPA_ONNX_LOGE("Setup pointers successfully.");
      } else {
        SHERPA_ONNX_LOGE("Bad things happened in setting up pointers.");
      }
    }
  }

 private:
  bool debug_ = true;
  std::unique_ptr<void, decltype(&dlclose)> model_lib_handle_{nullptr,
                                                              &dlclose};

  std::unique_ptr<void, decltype(&dlclose)> system_lib_handle_{nullptr,
                                                               &dlclose};

  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_;

  ComposeGraphsFnHandleType compose_graphs_fn_handle_ = nullptr;
  FreeGraphInfoFnHandleType free_graph_info_fn_handle_ = nullptr;

  std::vector<TensorPtr> input_tensors_;
  std::vector<TensorPtr> output_tensors_;

  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> output_tensor_names_;

  std::unordered_map<std::string, Qnn_Tensor_t *> name2tensor_;

  std::vector<uint8_t> buffer_;
  const QnnBackend *backend_ = nullptr;

  Qnn_GraphHandle_t graph_handle_ = nullptr;

  const QnnContext_Config_t **context_config_ = nullptr;
  bool is_initialized_ = false;
};

QnnModel::~QnnModel() = default;

QnnModel::QnnModel(const std::string &model_so, const QnnBackend *backend,
                   bool debug)
    : impl_(std::make_unique<Impl>(model_so, backend, debug)) {}

QnnModel::QnnModel(const std::string &binary_context_file,
                   const std::string &system_lib, const QnnBackend *backend,
                   BinaryContextTag tag, bool debug)
    : impl_(std::make_unique<Impl>(binary_context_file, system_lib, backend,
                                   tag, debug)) {}  // NOLINT

bool QnnModel::SaveBinaryContext(const std::string &filename) const {
  return impl_->SaveBinaryContext(filename);
}

const std::vector<std::string> &QnnModel::InputTensorNames() const {
  return impl_->InputTensorNames();
}

const std::vector<std::string> &QnnModel::OutputTensorNames() const {
  return impl_->OutputTensorNames();
}

std::vector<int32_t> QnnModel::TensorShape(const std::string &name) const {
  return impl_->TensorShape(name);
}

int32_t QnnModel::TensorSizeInBytes(const std::string &name) const {
  return impl_->TensorSizeInBytes(name);
}

bool QnnModel::HasTensor(const std::string &name) const {
  return impl_->HasTensor(name);
}

bool QnnModel::SetInputTensorData(const std::string &name, const float *p,
                                  int32_t n) const {
  return impl_->SetInputTensorData(name, p, n);
}

bool QnnModel::SetInputTensorData(const std::string &name, const int32_t *p,
                                  int32_t n) const {
  return impl_->SetInputTensorData(name, p, n);
}

std::vector<float> QnnModel::GetOutputTensorData(
    const std::string &name) const {
  return impl_->GetOutputTensorData(name);
}

bool QnnModel::Run() const { return impl_->Run(); }

bool QnnModel::IsInitialized() const { return impl_->IsInitialized(); }

}  // namespace sherpa_onnx
