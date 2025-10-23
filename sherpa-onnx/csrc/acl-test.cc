
#include "acl/acl.h"

#include <stdio.h>

#include <cstdint>

int main() {
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;

  aclError ret = aclInit(nullptr);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclInit: %s\n", msg);
    return -1;
  }

  printf("aclInit: Success\n");

  int32_t major, minor, patch;
  ret = aclrtGetVersion(&major, &minor, &patch);

  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtGetVersion: %s\n", msg);
    return -1;
  }

  int32_t device_id = 0;
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtSetDevice: %s\n", msg);
    return 1;
  }

  printf("aclrtSetDevice: device id %d\n", device_id);

  ret = aclrtCreateContext(&context, device_id);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtCreateContext: %s\n", msg);
    return 1;
  }
  printf("aclrtCreateContext: Success\n");

  ret = aclrtCreateStream(&stream);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtCreateStream: %s\n", msg);
    return 1;
  }

  printf("aclrtCreateStream: Success\n");

  ret = aclrtDestroyStream(stream);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtDestroyStream: %s\n", msg);
    return 1;
  }

  printf("aclrtDestroyStream: Success\n");

  ret = aclrtDestroyContext(context);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtDestroyContext: %s\n", msg);
    return 1;
  }

  printf("aclrtDestroyContext: Success\n");

  ret = aclrtResetDevice(device_id);
  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclrtResetDevice: %s\n", msg);
    return 1;
  }

  printf("aclrtResetDevice: Success\n");

  ret = aclFinalize();

  if (ret != ACL_ERROR_NONE) {
    const char *msg = aclGetRecentErrMsg();
    fprintf(stderr, "aclFinalize: %s\n", msg);
    return 1;
  }

  printf("aclFinalize: Success\n");

  return 0;
}
