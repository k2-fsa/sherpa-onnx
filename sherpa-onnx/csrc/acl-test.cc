
#include "acl/acl.h"

#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <vector>

void TestMemory() {
  std::vector<float> v = {1, 2, 3, 5.5};

  void *device_buffer = nullptr;
  auto ret = aclrtMalloc(&device_buffer, v.size() * sizeof(float),
                         ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtMalloc: %s\n", msg);
    exit(-1);
  }

  ret = aclrtMemcpy(device_buffer, v.size() * sizeof(float), v.data(),
                    v.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtMemcpy: %s\n", msg);
    exit(-1);
  }

  std::vector<float> v2;
  v2.resize(4);

  ret = aclrtMemcpy(v2.data(), v2.size() * sizeof(float), device_buffer,
                    v.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtMemcpy: %s\n", msg);
    exit(-1);
  }

  for (auto i : v2) {
    printf("%.3f ", i);
  }
  printf("\n");
}

void PrintModelInfo(aclmdlDesc *model_desc) {
  printf("---printing model info---\n");
  size_t num_inputs = aclmdlGetNumInputs(model_desc);
  size_t num_outputs = aclmdlGetNumOutputs(model_desc);
  printf("num inputs: %zu\n", num_inputs);
  printf("num outputs: %zu\n", num_outputs);

  for (int32_t i = 0; i < num_inputs; ++i) {
    printf("---input %d---\n", i);
    size_t size_in_bytes = aclmdlGetInputSizeByIndex(model_desc, i);

    printf(" size in bytes: %zu\n", size_in_bytes);
    printf(" size in MB:    %zu\n", size_in_bytes / 1024 / 1024);

    const char *name = aclmdlGetInputNameByIndex(model_desc, i);
    printf(" name: %s\n", name);

    aclFormat format = aclmdlGetInputFormat(model_desc, i);
    printf(" format: %d\n", format);  // 2 -> ACL_FORMAT_ND
    aclDataType type = aclmdlGetInputDataType(model_desc, i);
    printf(" data type: %d\n", type);  // 0 -> ACL_FLOAT
                                       //
    size_t k;
    aclError ret = aclmdlGetInputIndexByName(model_desc, name, &k);
    // check ret
    printf(" index: %zu\n", k);

    aclmdlIODims dims;
    ret = aclmdlGetInputDims(model_desc, i, &dims);
    printf(" dim: %zu\n", dims.dimCount);
    for (size_t d = 0; d < dims.dimCount; ++d) {
      printf("  %zu -> %s, %d\n", d, dims.name, (int)dims.dims[d]);
    }
  }
}

int test1() {
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;

  aclError ret = aclInit(nullptr);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclInit: %s\n", msg);
    exit(-1);
  }

  printf("aclInit: Success\n");

  int32_t major, minor, patch;
  ret = aclrtGetVersion(&major, &minor, &patch);

  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtGetVersion: %s\n", msg);
    exit(-1);
  }
  printf("version: %d.%d.%d\n", major, minor, patch);

  int32_t device_id = 0;
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtSetDevice: %s\n", msg);
    exit(-1);
  }

  printf("aclrtSetDevice: device id %d\n", device_id);

  size_t free, total;
  ret = aclrtGetMemInfo(ACL_DDR_MEM, &free, &total);

  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtGetMemInfo: %s\n", msg);
    exit(-1);
  }
  printf("DDR free: %.3f GB, total: %.3f GB\n", free / 1024. / 1024. / 1024,
         total / 1024. / 1024. / 1024.);

  ret = aclrtGetMemInfo(ACL_HBM_MEM, &free, &total);

  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtGetMemInfo: %s\n", msg);
    exit(-1);
  }
  printf("HBM free: %.3f GB, total: %.3f GB\n", free / 1024. / 1024. / 1024,
         total / 1024. / 1024. / 1024.);

  ret = aclrtCreateContext(&context, device_id);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtCreateContext: %s\n", msg);
    exit(-1);
  }
  printf("aclrtCreateContext: Success\n");

  TestMemory();

  printf("---loading model ./model.om\n");

  const char *model_path = "./model.om";  // Path to OM file
  uint32_t model_id;
  ret = aclmdlLoadFromFile(model_path, &model_id);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclmdlLoadFromFile: %s\n", msg);
    exit(-1);
  }

  printf("load model success\n");

  aclmdlDesc *model_desc = aclmdlCreateDesc();

  if (!model_desc) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclmdlCreateDesc: %s", msg);
    exit(-1);
  }

  ret = aclmdlGetDesc(model_desc, model_id);
  // check ret

  PrintModelInfo(model_desc);

  ret = aclmdlDestroyDesc(model_desc);
  // check ret

  ret = aclmdlUnload(model_id);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclmdlUnload: %s\n", msg);
    exit(-1);
  }

  printf("unload model success");

  ret = aclrtCreateStream(&stream);
  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtCreateStream: %s\n", msg);
    exit(-1);
  }

  printf("aclrtCreateStream: Success\n");

  ret = aclrtDestroyStream(stream);
  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtDestroyStream: %s\n", msg);
    exit(-1);
  }

  printf("aclrtDestroyStream: Success\n");

  ret = aclrtDestroyContext(context);
  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtDestroyContext: %s\n", msg);
    exit(-1);
  }

  printf("aclrtDestroyContext: Success\n");

  ret = aclrtResetDevice(device_id);
  if (ret != ACL_RT_SUCCESS) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtResetDevice: %s\n", msg);
    exit(-1);
  }

  printf("aclrtResetDevice: Success\n");

  ret = aclFinalize();

  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclFinalize: %s\n", msg);
    exit(-1);
  }

  printf("aclFinalize: Success\n");

  return 0;
}

int main() { test1(); }
