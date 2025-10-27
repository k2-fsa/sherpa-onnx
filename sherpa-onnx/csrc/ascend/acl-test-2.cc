#include <cstdio>
#include <vector>

#include "acl/acl.h"

// Helper: allocate device buffer and copy host data
void *createDeviceBuffer(float *hostData, size_t size) {
  void *deviceData = nullptr;
  aclError ret = aclrtMalloc(&deviceData, size, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != ACL_ERROR_NONE) {
    printf("aclrtMalloc failed: %d\n", ret);
    return nullptr;
  }
  ret =
      aclrtMemcpy(deviceData, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_ERROR_NONE) {
    printf("aclrtMemcpy failed: %d\n", ret);
    aclrtFree(deviceData);
    return nullptr;
  }
  return deviceData;
}

// Create input dataset
aclmdlDataset *createInputDataset(float *features, uint32_t timeLen,
                                  aclDataBuffer **outBuf) {
  size_t inputSize = 1 * timeLen * 512 * sizeof(float);
  void *deviceData = createDeviceBuffer(features, inputSize);
  if (!deviceData) return nullptr;

  aclDataBuffer *inputBuf = aclCreateDataBuffer(deviceData, inputSize);
  if (!inputBuf) {
    printf("aclCreateDataBuffer failed\n");
    aclrtFree(deviceData);
    return nullptr;
  }

  aclmdlDataset *dataset = aclmdlCreateDataset();
  if (!dataset) {
    printf("aclmdlCreateDataset failed\n");
    aclDestroyDataBuffer(inputBuf);
    aclrtFree(deviceData);
    return nullptr;
  }

  if (aclmdlAddDatasetBuffer(dataset, inputBuf) != ACL_ERROR_NONE) {
    printf("aclmdlAddDatasetBuffer failed\n");
    aclmdlDestroyDataset(dataset);
    aclDestroyDataBuffer(inputBuf);
    aclrtFree(deviceData);
    return nullptr;
  }

  *outBuf = inputBuf;
  return dataset;
}

int main() {
  const char *modelPath = "./model.om";
  uint32_t deviceId = 0;

  // -------------------- 1. Initialize ACL --------------------
  if (aclInit(nullptr) != ACL_ERROR_NONE) return -1;
  if (aclrtSetDevice(deviceId) != ACL_ERROR_NONE) return -1;

  aclrtContext context;
  if (aclrtCreateContext(&context, deviceId) != ACL_ERROR_NONE) return -1;
  if (aclrtSetCurrentContext(context) != ACL_ERROR_NONE) return -1;

  // -------------------- 2. Load model --------------------
  uint32_t model_id = 0;
  auto ret = aclmdlLoadFromFile(modelPath, &model_id);
  if (ret != ACL_ERROR_NONE) {
    printf("Failed to load model, aclmdlLoadFromFile ret=%d\n", ret);
    return -1;
  }
  printf("Model loaded successfully, modelId=0x%x\n", model_id);

  printf("Model loaded, modelId=0x%x\n", model_id);

  aclmdlDesc *modelDesc = aclmdlCreateDesc();
  if (aclmdlGetDesc(modelDesc, model_id) != ACL_ERROR_NONE) {
    printf("Failed to get model desc\n");
    return -1;
  }

  // -------------------- 3. Prepare dynamic input --------------------
  uint32_t timeLen = 100;  // example sequence length
  std::vector<float> features(1 * timeLen * 512, 0.0f);  // dummy data

  aclDataBuffer *inputBuf = nullptr;
  aclmdlDataset *inputDataset =
      createInputDataset(features.data(), timeLen, &inputBuf);
  if (!inputDataset) return -1;

  // -------------------- 4. Set dynamic dims --------------------
  aclmdlIODims dims;
  dims.dimCount = 3;
  dims.dims[0] = 1;        // batch
  dims.dims[1] = timeLen;  // actual sequence length
  dims.dims[2] = 512;      // feature dim

  if (aclmdlSetInputDynamicDims(model_id, inputDataset, 0, &dims) !=
      ACL_ERROR_NONE) {
    printf("aclmdlSetInputDynamicDims failed\n");
    return -1;
  }
  printf("here after set dynamic dims\n");
  return 0;

  // -------------------- 5. Prepare output dataset --------------------
  size_t outputSize =
      1 * timeLen * sizeof(float);  // replace with actual output size
  void *outputDev = nullptr;
  if (aclrtMalloc(&outputDev, outputSize, ACL_MEM_MALLOC_NORMAL_ONLY) !=
      ACL_ERROR_NONE)
    return -1;

  aclDataBuffer *outputBuf = aclCreateDataBuffer(outputDev, outputSize);
  aclmdlDataset *outputDataset = aclmdlCreateDataset();
  aclmdlAddDatasetBuffer(outputDataset, outputBuf);

  // -------------------- 6. Execute model --------------------
  if (aclmdlExecute(model_id, inputDataset, outputDataset) != ACL_ERROR_NONE) {
    printf("aclmdlExecute failed\n");
    return -1;
  }
  printf("Model executed successfully\n");

  // Optionally copy output back to host
  std::vector<float> hostOutput(timeLen);
  aclrtMemcpy(hostOutput.data(), outputSize, outputDev, outputSize,
              ACL_MEMCPY_DEVICE_TO_HOST);

  // -------------------- 7. Cleanup --------------------
  aclmdlDestroyDataset(inputDataset);
  aclmdlDestroyDataset(outputDataset);
  aclDestroyDataBuffer(inputBuf);
  aclDestroyDataBuffer(outputBuf);
  aclrtFree(outputDev);
  aclmdlDestroyDesc(modelDesc);
  aclmdlUnload(model_id);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
