
#include "acl/acl.h"

#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <vector>

#define CHECK(ret)                                                  \
  do {                                                              \
    if (ret != 0) {                                                 \
      const char *msg = aclGetRecentErrMsg();                       \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, (int)__LINE__, msg); \
      exit(-1);                                                     \
    }                                                               \
  } while (0)

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

  for (int32_t i = 0; i < num_outputs; ++i) {
    printf("---output %d---\n", i);
    size_t size_in_bytes = aclmdlGetOutputSizeByIndex(model_desc, i);

    printf(" size in bytes: %zu\n", size_in_bytes);
    printf(" size in MB:    %zu\n", size_in_bytes / 1024 / 1024);

    const char *name = aclmdlGetOutputNameByIndex(model_desc, i);
    printf(" name: %s\n", name);

    aclFormat format = aclmdlGetOutputFormat(model_desc, i);
    printf(" format: %d\n", format);  // 2 -> ACL_FORMAT_ND
    aclDataType type = aclmdlGetOutputDataType(model_desc, i);
    printf(" data type: %d\n", type);  // 0 -> ACL_FLOAT
                                       //
    size_t k;
    aclError ret = aclmdlGetOutputIndexByName(model_desc, name, &k);
    // check ret
    printf(" index: %zu\n", k);

    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(model_desc, i, &dims);
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
  CHECK(ret);

  printf("aclInit: Success\n");

  uint32_t count = 0;
  ret = aclrtGetDeviceCount(&count);
  CHECK(ret);
  printf("num devices: %u\n", count);

  int32_t device_id = 1;
  ret = aclrtSetDevice(device_id);
  CHECK(ret);

  ret = aclrtCreateContext(&context, device_id);
  CHECK(ret);

  ret = aclrtSetCurrentContext(context);
  CHECK(ret);

  void *ctx = nullptr;
  ret = aclrtGetCurrentContext(&ctx);
  CHECK(ret);
  printf("ctx: %p, context: %p\n", ctx, context);

  ret = aclrtCreateStream(&stream);
  CHECK(ret);

  printf("aclrtCreateStream: Success\n");

  aclrtRunMode run_mode;
  ret = aclrtGetRunMode(&run_mode);
  CHECK(ret);

  printf("run mode: %d, --> == host: %d, == device: %d\n", run_mode,
         run_mode == ACL_HOST, run_mode == ACL_DEVICE);

  printf("---loading model ./model.om\n");

  size_t work_size = 1, weight_size = 1;
  ret = aclmdlQuerySize("./model.om", &work_size, &weight_size);
  CHECK(ret);
  printf("work size: %zu (%x), %zu (%x)\n", work_size, work_size, weight_size,
         weight_size);

  void *model_work_ptr = nullptr;
  void *weight_ptr = nullptr;
  if (work_size > 0) {
    ret = aclrtMalloc(&model_work_ptr, work_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK(ret);
  }

  ret = aclrtMalloc(&weight_ptr, weight_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK(ret);

  uint32_t model_id = 0;
  ret = aclmdlLoadFromFileWithMem("./model.om", &model_id, model_work_ptr,
                                  work_size, weight_ptr, weight_size);
  CHECK(ret);

  fprintf(stderr,
          "loading model2: %d, model2 "
          "id: %u %x\n",
          (int)ret, model_id, model_id);

  printf("size of size_t: %zu\n", sizeof(size_t));

  printf("after model id: %u, %x\n", model_id, model_id);

  printf("load model success\n");

  aclmdlDesc *model_desc = aclmdlCreateDesc();

  if (!model_desc) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclmdlCreateDesc: %s", msg);
    exit(-1);
  }

  ret = aclmdlGetDesc(model_desc, model_id);
  CHECK(ret);

  PrintModelInfo(model_desc);

  std::vector<float> features(83 * 512);
  void *input_dev = nullptr;
  size_t input_size = features.size() * sizeof(float);

  ret = aclrtMalloc(&input_dev, input_size, ACL_MEM_MALLOC_NORMAL_ONLY);
  CHECK(ret);
  printf("input dev: %p\n", input_dev);

  ret = aclrtMemcpy(input_dev, input_size, features.data(), input_size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK(ret);
  // check ret
  aclmdlDataset *input_dataset = aclmdlCreateDataset();
  aclDataBuffer *input_buf = aclCreateDataBuffer(input_dev, input_size);
  printf("dataset: %p, buf: %p\n", input_dataset, input_buf);

  ret = aclmdlAddDatasetBuffer(input_dataset, input_buf);
  CHECK(ret);

  aclmdlIODims dims;
  // ret = aclmdlGetInputDims(model_desc, 0,
  // &dims); CHECK(ret);

  dims.dimCount = 3;
  dims.dims[0] = 1;
  dims.dims[1] = 83;
  dims.dims[2] = 512;

  printf("here mem-->%d, %d\n",
         (int32_t)(dims.dims[0] * dims.dims[1] * dims.dims[2]) * 4,
         (int)input_size);

  // dims.dims[1] = num_frames;
  printf("%d, %d, %d\n", (int)dims.dims[0], (int)dims.dims[1],
         (int)dims.dims[2]);

  printf("DEBUG: modelId = 0x%x (%u)\n", model_id, model_id);

  size_t gear_count = 0;
  ret = aclmdlGetInputDynamicGearCount(model_desc, -1, &gear_count);
  CHECK(ret);
  printf("gear count: %d\n", (int)gear_count);

  size_t index = 0;
  // ret = aclmdlGetInputIndexByName(model_desc, ACL_DYNAMIC_TENSOR_NAME,
  // &index);
  ret = aclmdlGetInputIndexByName(model_desc, "encoder_out", &index);
  CHECK(ret);

  printf("dynamic index: %zu\n", index);

  aclrtContext curCtx = NULL;
  ret = aclrtGetCurrentContext(&curCtx);
  CHECK(ret);
  printf("DEBUG: current context = %p, %p\n", curCtx, context);

  printf("DEBUG: dataset: %p, buf: %p\n", (void *)input_dataset,
         (void *)input_buf);

  // dims you're about to set
  printf(
      "DEBUG: setting dims rank=%zu [%ld, "
      "%ld, %ld]\n",
      dims.dimCount, dims.dims[0], dims.dims[1], dims.dims[2]);

  ret = aclmdlSetInputDynamicDims(model_id, input_dataset, 0, &dims);
  CHECK(ret);
  printf(
      "DEBUG: aclmdlSetInputDynamicDims "
      "returned %d\n",
      ret);

  // check ret
  size_t size_in_bytes = aclmdlGetInputSizeByIndex(model_desc, 0);
  size_t size_out_bytes = aclmdlGetOutputSizeByIndex(model_desc, 0);
  printf("in bytes: %d, out_bytes: %d\n", (int)size_in_bytes,
         (int)size_out_bytes);
  PrintModelInfo(model_desc);

  ret = aclmdlDestroyDesc(model_desc);
  CHECK(ret);

  ret = aclmdlUnload(model_id);
  CHECK(ret);

  printf("unload model success\n");

  ret = aclrtDestroyStream(stream);
  CHECK(ret);

  printf("aclrtDestroyStream: Success\n");

  ret = aclrtDestroyContext(context);
  CHECK(ret);

  printf("aclrtDestroyContext: Success\n");

  ret = aclrtResetDevice(device_id);
  CHECK(ret);

  printf("aclrtResetDevice: Success\n");

  ret = aclFinalize();
  CHECK(ret);

  printf("aclFinalize: Success\n");

  return 0;
}

int main() { test1(); }
