// sherpa-onnx/csrc/ascend/utils.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/ascend/utils.h"

#include <sstream>

#include "sherpa-onnx/csrc/ascend/macros.h"

namespace sherpa_onnx {

static const char *AclDataTypeToString(aclDataType data_type) {
  switch (data_type) {
    case ACL_DT_UNDEFINED:
      return "ACL_DT_UNDEFINED";
    case ACL_FLOAT:
      return "ACL_FLOAT";
    case ACL_FLOAT16:
      return "ACL_FLOAT16";
    case ACL_INT8:
      return "ACL_INT8";
    case ACL_INT32:
      return "ACL_INT32";
    case ACL_UINT8:
      return "ACL_UINT8";
    case ACL_INT16:
      return "ACL_INT16";
    case ACL_UINT16:
      return "ACL_UINT16";
    case ACL_UINT32:
      return "ACL_UINT32";
    case ACL_INT64:
      return "ACL_INT64";
    case ACL_UINT64:
      return "ACL_UINT64";
    case ACL_DOUBLE:
      return "ACL_DOUBLE";
    case ACL_BOOL:
      return "ACL_BOOL";
    case ACL_STRING:
      return "ACL_STRING";
    case ACL_COMPLEX64:
      return "ACL_COMPLEX64";
    case ACL_COMPLEX128:
      return "ACL_COMPLEX128";
    case ACL_BF16:
      return "ACL_BF16";
    case ACL_INT4:
      return "ACL_INT4";
    case ACL_UINT1:
      return "ACL_UINT1";
    case ACL_COMPLEX32:
      return "ACL_COMPLEX32";
    default:
      return "unknown";
  }
}

static const char *AclFormatToString(aclFormat format) {
  switch (format) {
    case ACL_FORMAT_UNDEFINED:
      return "ACL_FORMAT_UNDEFINED";
    case ACL_FORMAT_NCHW:
      return "ACL_FORMAT_NCHW";
    case ACL_FORMAT_NHWC:
      return "ACL_FORMAT_NHWC";
    case ACL_FORMAT_ND:
      return "ACL_FORMAT_ND";
    case ACL_FORMAT_NC1HWC0:
      return "ACL_FORMAT_NC1HWC0";
    case ACL_FORMAT_FRACTAL_Z:
      return "ACL_FORMAT_FRACTAL_Z";
    case ACL_FORMAT_NC1HWC0_C04:
      return "ACL_FORMAT_NC1HWC0_C04";
    case ACL_FORMAT_HWCN:
      return "ACL_FORMAT_HWCN";
    case ACL_FORMAT_NDHWC:
      return "ACL_FORMAT_NDHWC";
    case ACL_FORMAT_FRACTAL_NZ:
      return "ACL_FORMAT_FRACTAL_NZ";
    case ACL_FORMAT_NCDHW:
      return "ACL_FORMAT_NCDHW";
    case ACL_FORMAT_NDC1HWC0:
      return "ACL_FORMAT_NDC1HWC0";
    case ACL_FRACTAL_Z_3D:
      return "ACL_FRACTAL_Z_3D";
    case ACL_FORMAT_NC:
      return "ACL_FORMAT_NC";
    case ACL_FORMAT_NCL:
      return "ACL_FORMAT_NCL";
    default:
      return "unknown";
  }
}

Acl::Acl() {
  aclError ret = aclInit(nullptr);
  SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclInit");
  initialized_ = true;
}

Acl::~Acl() {
  if (initialized_) {
    aclError ret = aclFinalize();
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclFinalize");
  }
}

AclContext::AclContext(int32_t device_id) {
  aclError ret = aclrtCreateContext(&context_, device_id);
  SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtCreateContext");
}

AclContext::~AclContext() {
  if (context_) {
    aclError ret = aclrtDestroyContext(context_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtDestroyContext");
  }
}

aclrtContext AclContext::Get() const { return context_; }

AclDevicePtr::AclDevicePtr(
    size_t size, aclrtMemMallocPolicy policy /*= ACL_MEM_MALLOC_HUGE_FIRST*/) {
  if (size > 0) {
    aclError ret = aclrtMalloc(&p_, size, policy);

    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtMalloc with size: %zu",
                             size);
  }
  size_ = size;
}

AclDevicePtr::~AclDevicePtr() {
  if (p_) {
    aclError ret = aclrtFree(p_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtFree");
  }
}

AclHostPtr::AclHostPtr(size_t size) {
  if (size > 0) {
    aclError ret = aclrtMallocHost(&p_, size);

    SHERPA_ONNX_ASCEND_CHECK(
        ret, "Failed to call aclrtMallocHost with size: %zu", size);
  }
  size_ = size;
}

AclHostPtr::~AclHostPtr() {
  if (p_) {
    aclError ret = aclrtFreeHost(p_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclrtFreeHost");
  }
}

AclModelDesc::AclModelDesc(uint32_t model_id) {
  p_ = aclmdlCreateDesc();
  if (!p_) {
    SHERPA_ONNX_LOGE("Failed to call aclmdlCreateDesc");
    SHERPA_ONNX_EXIT(-1);
  }

  aclError ret = aclmdlGetDesc(p_, model_id);
  SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlGetDesc");
}

AclModelDesc::~AclModelDesc() {
  if (p_) {
    aclError ret = aclmdlDestroyDesc(p_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlDestroyDesc");
  }
}

AclModel::AclModel(const std::string &model_path) {
  aclError ret = aclmdlLoadFromFile(model_path.c_str(), &model_id_);

  SHERPA_ONNX_ASCEND_CHECK(ret,
                           "Failed to call aclmdlLoadFromFile from file '%s'",
                           model_path.c_str());

  desc_ = std::make_unique<AclModelDesc>(model_id_);
}

AclModel::~AclModel() {
  if (model_id_ != 0) {
    aclError ret = aclmdlUnload(model_id_);
    SHERPA_ONNX_ASCEND_CHECK(ret, "Failed to call aclmdlUnload");
  }
}

std::string AclModel::GetInfo() const {
  size_t num_inputs = aclmdlGetNumInputs(desc_->Get());
  size_t num_outputs = aclmdlGetNumOutputs(desc_->Get());

  std::ostringstream os;
  os << "Model id: " << model_id_ << "\n";
  os << "Num inputs: " << num_inputs << "\n";
  os << "Num outputs: " << num_outputs << "\n";

  for (int32_t i = 0; i < num_inputs; ++i) {
    os << "---input " << i << "---\n";

    size_t size_in_bytes = aclmdlGetInputSizeByIndex(desc_->Get(), i);

    os << " size in bytes: " << size_in_bytes << "\n";
    os << " size in MB:    " << size_in_bytes / 1024 / 1024 << "\n";

    const char *name = aclmdlGetInputNameByIndex(desc_->Get(), i);
    os << " name: " << name << "\n";

    aclFormat format = aclmdlGetInputFormat(desc_->Get(), i);

    os << " format: " << AclFormatToString(format) << "\n";
    aclDataType type = aclmdlGetInputDataType(desc_->Get(), i);
    os << " data type: " << AclDataTypeToString(type) << "\n";

    aclmdlIODims dims;
    aclError ret = aclmdlGetInputDims(desc_->Get(), i, &dims);
    os << " dim: " << dims.dimCount << "\n";
    for (size_t d = 0; d < dims.dimCount; ++d) {
      os << "  " << d << " -> " << dims.name << ", " << dims.dims[d] << "\n";
    }
  }

  for (int32_t i = 0; i < num_outputs; ++i) {
    os << "---output " << i << "---\n";

    size_t size_out_bytes = aclmdlGetOutputSizeByIndex(desc_->Get(), i);

    os << " size out bytes: " << size_out_bytes << "\n";
    os << " size out MB:    " << size_out_bytes / 1024 / 1024 << "\n";

    const char *name = aclmdlGetOutputNameByIndex(desc_->Get(), i);
    os << " name: " << name << "\n";

    aclFormat format = aclmdlGetOutputFormat(desc_->Get(), i);

    os << " format: " << AclFormatToString(format) << "\n";
    aclDataType type = aclmdlGetInputDataType(desc_->Get(), i);
    os << " data type: " << AclDataTypeToString(type) << "\n";

    aclmdlIODims dims;
    aclError ret = aclmdlGetOutputDims(desc_->Get(), i, &dims);
    os << " dim: " << dims.dimCount << "\n";
    for (size_t d = 0; d < dims.dimCount; ++d) {
      os << "  " << d << " -> " << dims.name << ", " << dims.dims[d] << "\n";
    }
  }

  return os.str();
}

}  // namespace sherpa_onnx
