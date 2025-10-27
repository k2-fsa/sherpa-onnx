// sherpa-onnx/csrc/ascend/utils.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ASCEND_UTILS_H_
#define SHERPA_ONNX_CSRC_ASCEND_UTILS_H_

#include <memory>
#include <string>
#include <vector>

#include "acl/acl.h"

namespace sherpa_onnx {

class Acl {
 public:
  Acl();
  ~Acl();

  Acl(const Acl &) = delete;
  const Acl &operator=(const Acl &) = delete;

 private:
  bool initialized_ = false;
};

class AclContext {
 public:
  explicit AclContext(int32_t device_id);
  explicit AclContext(aclrtContext context) : context_(context) {}

  ~AclContext();

  AclContext(const AclContext &) = delete;
  const AclContext &operator=(const AclContext &) = delete;

  aclrtContext Get() const;
  operator aclrtContext() { return context_; }

 private:
  aclrtContext context_ = nullptr;
};

class AclDevicePtr {
 public:
  explicit AclDevicePtr(
      size_t size, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);

  ~AclDevicePtr();

  AclDevicePtr(const AclDevicePtr &) = delete;
  const AclDevicePtr &operator=(const AclDevicePtr &) = delete;

  void *Get() const { return p_; }
  operator void *() { return p_; }

  size_t Size() const { return size_; }

 private:
  void *p_ = nullptr;
  size_t size_ = 0;
};

class AclHostPtr {
 public:
  explicit AclHostPtr(size_t size);

  ~AclHostPtr();

  AclHostPtr(const AclDevicePtr &) = delete;
  const AclHostPtr &operator=(const AclDevicePtr &) = delete;

  void *Get() const { return p_; }
  operator void *() { return p_; }

 private:
  void *p_ = nullptr;
  size_t size_ = 0;
};

class AclModelDesc {
 public:
  explicit AclModelDesc(uint32_t model_id);

  ~AclModelDesc();

  AclModelDesc(const AclModelDesc &) = delete;
  const AclModelDesc &operator=(const AclModelDesc &) = delete;

  aclmdlDesc *Get() const { return p_; }
  operator aclmdlDesc *() const { return p_; }

  size_t Size() const { return size_; }

 private:
  aclmdlDesc *p_ = nullptr;
  size_t size_ = 0;
};

class AclModel {
 public:
  explicit AclModel(const std::string &model_path);
  AclModel(const void *model, size_t model_size);
  ~AclModel();

  uint32_t Get() const { return model_id_; }
  operator uint32_t() const { return model_id_; }

  AclModel(const AclModel &) = delete;
  const AclModel &operator=(const AclModel &) = delete;

  std::string GetInfo() const;

  const std::vector<std::string> &GetInputNames() const { return input_names_; }

  const std::vector<std::vector<int64_t>> &GetInputShapes() const {
    return input_shapes_;
  }

  const std::vector<std::string> &GetOutputNames() const {
    return output_names_;
  }

  const std::vector<std::vector<int64_t>> &GetOutputShapes() const {
    return output_shapes_;
  }

 private:
  void Init();
  void InitInputNames();
  void InitInputShapes();

  void InitOutputNames();
  void InitOutputShapes();

 private:
  uint32_t model_id_ = 0;
  std::unique_ptr<AclModelDesc> desc_;

  std::vector<std::string> input_names_;
  std::vector<std::vector<int64_t>> input_shapes_;

  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> output_shapes_;
};

class AclMdlDataset {
 public:
  AclMdlDataset();
  ~AclMdlDataset();

  AclMdlDataset(const AclMdlDataset &) = delete;
  AclMdlDataset &operator=(const AclMdlDataset &) = delete;

  void AddBuffer(aclDataBuffer *buffer) const;
  void SetTensorDesc(aclTensorDesc *tensor_desc, size_t index) const;

  aclmdlDataset *Get() const { return p_; }
  operator aclmdlDataset *() const { return p_; }

 private:
  aclmdlDataset *p_ = nullptr;
};

class AclDataBuffer {
 public:
  AclDataBuffer(void *data, size_t size);
  ~AclDataBuffer();

  AclDataBuffer(const AclDataBuffer &) = delete;
  AclDataBuffer &operator=(const AclDataBuffer &) = delete;

  aclDataBuffer *Get() const { return p_; }
  operator aclDataBuffer *() const { return p_; }

 private:
  aclDataBuffer *p_ = nullptr;
};

class AclTensorDesc {
 public:
  AclTensorDesc(aclDataType data_type, int num_dims, const int64_t *dims,
                aclFormat format);
  ~AclTensorDesc();

  AclTensorDesc(const AclTensorDesc &) = delete;
  AclTensorDesc &operator=(const AclTensorDesc &) = delete;

  aclTensorDesc *Get() const { return p_; }
  operator aclTensorDesc *() const { return p_; }

 private:
  aclTensorDesc *p_ = nullptr;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_UTILS_H_
