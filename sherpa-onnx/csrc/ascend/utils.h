// sherpa-onnx/csrc/ascend/utils.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ASCEND_UTILS_H_
#define SHERPA_ONNX_CSRC_ASCEND_UTILS_H_

#include <memory>

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

 private:
  void *p_ = nullptr;
};

class AclHostPtr {
 public:
  explicit AclHostPtr(size_t size);

  ~AclHostPtr();

  AclHostPtr(const AclDevicePtr &) = delete;
  const AclHostPtr &operator=(const AclDevicePtr &) = delete;

  void *Get() const { return p_; }

 private:
  void *p_ = nullptr;
};

class AclModelDesc {
 public:
  explicit AclModelDesc(uint32_t model_id);

  ~AclModelDesc();

  AclModelDesc(const AclModelDesc &) = delete;
  const AclModelDesc &operator=(const AclModelDesc &) = delete;

  aclmdlDesc *Get() const { return p_; }

 private:
  aclmdlDesc *p_ = nullptr;
};

class AclModel {
 public:
  explicit AclModel(const std::string &model_path);
  ~AclModel();

  uint32_t Get() const { return model_id_; }

  AclModel(const AclModel &) = delete;
  const AclModel &operator=(const AclModel &) = delete;

  std::string GetInfo() const;

 private:
  uint32_t model_id_ = 0;
  std::unique_ptr<AclModelDesc> desc_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ASCEND_UTILS_H_
