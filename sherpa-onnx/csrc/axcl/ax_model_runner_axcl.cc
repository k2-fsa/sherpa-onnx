// sherpa-onnx/csrc/axcl/ax_model_runner_axcl.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "ax_model_runner_axcl.hpp"

#include <axcl.h>
#include <fcntl.h>
#include <string.h>

#include <fstream>
#include <memory>

typedef enum {
  AX_ENGINE_ABST_DEFAULT = 0,
  AX_ENGINE_ABST_CACHED = 1,
} AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

typedef std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T,
                  AX_ENGINE_ALLOC_BUFFER_STRATEGY_T>
    INPUT_OUTPUT_ALLOC_STRATEGY;

static void print_io_info(std::vector<ax_runner_tensor_t> &input,
                          std::vector<ax_runner_tensor_t> &output) {
  printf("\ninput size: %ld\n", input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    // print shape info,like [batchsize x channel x height x width]
    auto &info = input[i];
    printf("    name: \e[1;32m%8s \e[0m\n        \e[1;31m", info.sName.c_str());
    for (size_t s = 0; s < info.vShape.size(); s++) {
      printf("%d", info.vShape[s]);
      if (s != info.vShape.size() - 1) {
        printf(" x ");
      }
    }
    printf("\e[0m\n\n");
  }

  printf("\noutput size: %ld\n", output.size());
  for (size_t i = 0; i < output.size(); ++i) {
    // print shape info,like [batchsize x channel x height x width]
    auto &info = output[i];
    printf("    name: \e[1;32m%8s \e[0m\n        \e[1;31m", info.sName.c_str());
    for (size_t s = 0; s < info.vShape.size(); s++) {
      printf("%d", info.vShape[s]);
      if (s != info.vShape.size() - 1) {
        printf(" x ");
      }
    }
    printf("\e[0m\n\n");
  }
}

static bool read_file(const char *fn, std::vector<unsigned char> &data) {
  FILE *fp = fopen(fn, "r");
  if (fp != nullptr) {
    fseek(fp, 0L, SEEK_END);
    auto len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    data.clear();
    size_t read_size = 0;
    if (len > 0) {
      data.resize(len);
      read_size = fread(data.data(), 1, len, fp);
    }
    fclose(fp);
    return read_size == (size_t)len;
  }
  return false;
}

typedef struct {
  int nIndex;
  int nSize;
  void *pBuf;
  void *pVirAddr;

  std::string Name;

  axclrtEngineIODims dims;
} AXCL_IO_BUF_T;

typedef struct {
  uint32_t nInputSize;
  uint32_t nOutputSize;
  AXCL_IO_BUF_T *pInputs;
  AXCL_IO_BUF_T *pOutputs;
} AXCL_IO_DATA_T;

static void free_io_index(AXCL_IO_BUF_T *pBuf, size_t index) {
  for (size_t i = 0; i < index; ++i) {
    axclrtFree(pBuf[i].pBuf);
  }
}

static void free_io(AXCL_IO_DATA_T *io_data) {
  for (size_t j = 0; j < io_data->nInputSize; ++j) {
    axclrtFree(io_data->pInputs[j].pBuf);
    free(io_data->pInputs[j].pVirAddr);
  }
  for (size_t j = 0; j < io_data->nOutputSize; ++j) {
    axclrtFree(io_data->pOutputs[j].pBuf);
    free(io_data->pOutputs[j].pVirAddr);
  }
  delete[] io_data->pInputs;
  delete[] io_data->pOutputs;
}

static inline int prepare_io(int grpid, axclrtEngineIOInfo io_info,
                             axclrtEngineIO io, AXCL_IO_DATA_T *io_data,
                             INPUT_OUTPUT_ALLOC_STRATEGY strategy) {
  memset(io_data, 0, sizeof(AXCL_IO_DATA_T));

  auto inputNum = axclrtEngineGetNumInputs(io_info);
  auto outputNum = axclrtEngineGetNumOutputs(io_info);
  io_data->nInputSize = inputNum;
  io_data->nOutputSize = outputNum;
  io_data->pInputs = new AXCL_IO_BUF_T[inputNum];
  io_data->pOutputs = new AXCL_IO_BUF_T[outputNum];

  // 1. alloc inputs
  for (uint32_t i = 0; i < inputNum; i++) {
    auto bufSize = axclrtEngineGetInputSizeByIndex(io_info, grpid, i);
    void *devPtr = nullptr;
    axclError ret = 0;
    if (AX_ENGINE_ABST_DEFAULT == strategy.first) {
      ret = axclrtMalloc(&devPtr, bufSize,
                         axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);
    } else {
      ret = axclrtMallocCached(
          &devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);
    }

    if (ret != 0) {
      free_io_index(io_data->pInputs, i);
      fprintf(stderr, "Malloc input(index: %d, size: %ld) failed! ret=0x%x\n",
              i, bufSize, ret);
      return -1;
    }
    std::vector<char> tmp(bufSize, 0);
    axclrtMemcpy(devPtr, tmp.data(), bufSize,
                 axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE);
    // axclrtMemset(devPtr, 0, bufSize);

    axclrtEngineIODims dims;
    ret = axclrtEngineGetInputDims(io_info, grpid, i, &dims);
    if (ret != 0) {
      free_io_index(io_data->pInputs, i);
      fprintf(stderr, "Get input dims(index: %d) failed! ret=0x%x\n", i, ret);
      return -1;
    }

    io_data->pInputs[i].nIndex = i;
    io_data->pInputs[i].nSize = bufSize;
    io_data->pInputs[i].pBuf = devPtr;
    io_data->pInputs[i].dims = dims;
    io_data->pInputs[i].Name = axclrtEngineGetInputNameByIndex(io_info, i);
    io_data->pInputs[i].pVirAddr = malloc(bufSize);
    memset(io_data->pInputs[i].pVirAddr, 0, bufSize);
    ret = axclrtEngineSetInputBufferByIndex(io, i, devPtr, bufSize);
    if (ret != 0) {
      free_io_index(io_data->pInputs, i);
      fprintf(stderr,
              "Set input buffer(index: %d, size: %lu) failed! ret=0x%x\n", i,
              bufSize, ret);
      return -1;
    }
  }

  // 2. alloc outputs
  for (uint32_t i = 0; i < outputNum; i++) {
    auto bufSize = axclrtEngineGetOutputSizeByIndex(io_info, grpid, i);
    void *devPtr = NULL;
    axclError ret = 0;
    if (AX_ENGINE_ABST_DEFAULT == strategy.first) {
      ret = axclrtMalloc(&devPtr, bufSize,
                         axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);
    } else {
      ret = axclrtMallocCached(
          &devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST);
    }

    if (ret != 0) {
      free_io_index(io_data->pOutputs, i);
      fprintf(stderr, "Malloc output(index: %d, size: %ld) failed! ret=0x%x\n",
              i, bufSize, ret);
      return -1;
    }
    std::vector<char> tmp(bufSize, 0);
    axclrtMemcpy(devPtr, tmp.data(), bufSize,
                 axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE);
    axclrtEngineIODims dims;
    ret = axclrtEngineGetOutputDims(io_info, grpid, i, &dims);
    if (ret != 0) {
      free_io_index(io_data->pOutputs, i);
      fprintf(stderr, "Get output dims(index: %d) failed! ret=0x%x\n", i, ret);
      return -1;
    }

    io_data->pOutputs[i].nIndex = i;
    io_data->pOutputs[i].nSize = bufSize;
    io_data->pOutputs[i].pBuf = devPtr;
    io_data->pOutputs[i].dims = dims;
    io_data->pOutputs[i].Name = axclrtEngineGetOutputNameByIndex(io_info, i);
    io_data->pOutputs[i].pVirAddr = malloc(bufSize);
    memset(io_data->pOutputs[i].pVirAddr, 0, bufSize);
    ret = axclrtEngineSetOutputBufferByIndex(io, i, devPtr, bufSize);
    if (ret != 0) {
      free_io_index(io_data->pOutputs, i);
      fprintf(stderr,
              "Set output buffer(index: %d, size: %lu) failed! ret=0x%x\n", i,
              bufSize, ret);
      return -1;
    }
  }

  return 0;
}

struct ax_joint_runner_axcl_handle_t {
  uint64_t handle = 0;
  uint64_t context = 0;
  axclrtEngineIOInfo io_info = 0;
  std::vector<axclrtEngineIO> ios;
  std::vector<AXCL_IO_DATA_T> io_datas;

  // int algo_width, algo_height;
  // int algo_colorformat;
};

int ax_runner_axcl::sub_init() {
  // 4. create context
  int ret = axclrtEngineCreateContext(m_handle->handle, &m_handle->context);
  if (0 != ret) {
    fprintf(stderr, "axclrtEngineCreateContext failed.\n");
    return ret;
  }
  fprintf(stdout, "axclrtEngineCreateContextt is done. \n");

  // 5. set io

  ret = axclrtEngineGetIOInfo(m_handle->handle, &m_handle->io_info);
  if (0 != ret) {
    fprintf(stderr, "axclrtEngineGetIOInfo failed.\n");
    return ret;
  }
  fprintf(stdout, "axclrtEngineGetIOInfo is done. \n");

  ret = axclrtEngineGetShapeGroupsCount(m_handle->io_info, &group_count);
  if (ret != 0) {
    axclrtEngineUnload(m_handle->handle);
    return ret;
  }

  // 6. alloc io
  if (!_parepare_io) {
    m_handle->ios.resize(group_count);
    m_handle->io_datas.resize(group_count);
    mgroup_input_tensors.resize(group_count);
    mgroup_output_tensors.resize(group_count);

    memset(&m_handle->io_datas[0], 0, sizeof(AXCL_IO_DATA_T) * group_count);

    auto malloc_strategy =
        std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_DEFAULT);

    for (int grpid = 0; grpid < group_count; grpid++) {
      ret = axclrtEngineCreateIO(m_handle->io_info, &m_handle->ios[grpid]);
      if (ret != 0) {
        axclrtEngineUnload(m_handle->handle);
        fprintf(stderr, "Create io failed. ret=0x%x\n", ret);
        return -1;
      }

      ret = prepare_io(grpid, m_handle->io_info, m_handle->ios[grpid],
                       &m_handle->io_datas[grpid], malloc_strategy);
      if (ret != 0) {
        free_io(&m_handle->io_datas[grpid]);
        axclrtEngineDestroyIO(m_handle->ios[grpid]);
        axclrtEngineUnload(m_handle->handle);

        fprintf(stderr, "prepare_io failed.\n");
        return ret;
      }
    }

    for (int grpid = 0; grpid < group_count; grpid++) {
      // auto &io_info = m_handle->io_info[grpid];
      auto &io_data = m_handle->io_datas[grpid];
      for (uint32_t i = 0; i < io_data.nOutputSize; i++) {
        ax_runner_tensor_t tensor;
        tensor.nIdx = i;
        tensor.sName = std::string(io_data.pOutputs[i].Name);
        tensor.nSize = io_data.pOutputs[i].nSize;
        for (int32_t j = 0; j < io_data.pOutputs[i].dims.dimCount; j++) {
          tensor.vShape.push_back(io_data.pOutputs[i].dims.dims[j]);
        }
        tensor.phyAddr = (unsigned long long)io_data.pOutputs[i].pBuf;
        tensor.pVirAddr = io_data.pOutputs[i].pVirAddr;
        mgroup_output_tensors[grpid].push_back(tensor);
      }

      for (size_t i = 0; i < io_data.nInputSize; i++) {
        ax_runner_tensor_t tensor;
        tensor.nIdx = i;
        tensor.sName = std::string(io_data.pInputs[i].Name);
        tensor.nSize = io_data.pInputs[i].nSize;
        for (int32_t j = 0; j < io_data.pInputs[i].dims.dimCount; j++) {
          tensor.vShape.push_back(io_data.pInputs[i].dims.dims[j]);
        }
        tensor.phyAddr = (unsigned long long)io_data.pInputs[i].pBuf;
        tensor.pVirAddr = io_data.pInputs[i].pVirAddr;
        mgroup_input_tensors[grpid].push_back(tensor);
      }
    }

    moutput_tensors = mgroup_output_tensors[0];
    minput_tensors = mgroup_input_tensors[0];
    _parepare_io = true;
  } else {
  }
  // for (int grpid = 0; grpid < group_count; grpid++) {
  //   printf("\ngrpid: %d\n", grpid);
  //   print_io_info(mgroup_input_tensors[grpid], mgroup_output_tensors[grpid]);
  //   printf("==================================================\n\n");
  // }

  return ret;
}

int ax_runner_axcl::init(const char *model_file) {
  std::vector<unsigned char> model_buffer;
  if (!read_file(model_file, model_buffer)) {
    fprintf(stderr, "read_file failed.\n");
    return -1;
  }
  auto ret = init((char *)model_buffer.data(), model_buffer.size());
  return ret;
}

int ax_runner_axcl::init(char *model_buffer, size_t model_size) {
  if (!m_handle) {
    m_handle = new ax_joint_runner_axcl_handle_t;
  }
  memset((void *)m_handle, 0, sizeof(ax_joint_runner_axcl_handle_t));

  // 3. create handle
  void *devMem = nullptr;
  axclrtMalloc(&devMem, model_size, AXCL_MEM_MALLOC_NORMAL_ONLY);

  // 4. copy model to device
  axclrtMemcpy(devMem, model_buffer, model_size, AXCL_MEMCPY_HOST_TO_DEVICE);

  int ret = axclrtEngineLoadFromMem(devMem, model_size, &m_handle->handle);
  if (0 != ret) {
    fprintf(stderr, "AX_ENGINE_CreateHandle");
    return ret;
  }
  axclrtFree(devMem);

  return sub_init();
}

void ax_runner_axcl::release() {
  if (m_handle && m_handle->handle) {
    for (int grpid = 0; grpid < group_count; grpid++) {
      free_io(&m_handle->io_datas[grpid]);
      axclrtEngineDestroyIO(m_handle->ios[grpid]);
    }

    axclrtEngineUnload(m_handle->handle);
    m_handle->handle = 0;
  }

  if (m_handle) {
    delete m_handle;
    m_handle = nullptr;
  }

  minput_tensors.clear();
  moutput_tensors.clear();

  map_input_tensors.clear();
  map_output_tensors.clear();

  mgroup_input_tensors.clear();
  mgroup_output_tensors.clear();

  map_group_input_tensors.clear();
  map_group_output_tensors.clear();
}

void ax_runner_axcl::deinit() {
  if (m_handle && m_handle->handle) {
    axclrtEngineUnload(m_handle->handle);
    m_handle->handle = 0;
  }
}

int ax_runner_axcl::get_algo_width() {
  if (minput_tensors.size() == 1 && minput_tensors[0].vShape.size() == 4) {
    return minput_tensors[0].vShape[2];
  }
  return -1;
}
int ax_runner_axcl::get_algo_height() {
  if (minput_tensors.size() == 1 && minput_tensors[0].vShape.size() == 4) {
    return minput_tensors[0].vShape[1];
  }
  return -1;
}

int ax_runner_axcl::set_input(int grpid, int idx,
                              unsigned long long int phy_addr,
                              unsigned long size) {
  return axclrtEngineSetInputBufferByIndex(m_handle->ios[grpid], idx,
                                           (void *)phy_addr, size);
}
int ax_runner_axcl::set_output(int grpid, int idx,
                               unsigned long long int phy_addr,
                               unsigned long size) {
  return axclrtEngineSetOutputBufferByIndex(m_handle->ios[grpid], idx,
                                            (void *)phy_addr, size);
}

int ax_runner_axcl::set_input(int grpid, std::string name,
                              unsigned long long int phy_addr,
                              unsigned long size) {
  return axclrtEngineSetInputBufferByIndex(m_handle->ios[grpid],
                                           get_input(grpid, name).nIdx,
                                           (void *)phy_addr, size);
}

int ax_runner_axcl::set_output(int grpid, std::string name,
                               unsigned long long int phy_addr,
                               unsigned long size) {
  return axclrtEngineSetOutputBufferByIndex(m_handle->ios[grpid],
                                            get_output(grpid, name).nIdx,
                                            (void *)phy_addr, size);
}

ax_color_space_e ax_runner_axcl::get_color_space() {
  return ax_color_space_unknown;
}

int ax_runner_axcl::inference() { return inference(0); }

int ax_runner_axcl::inference(int grpid) {
  if (_auto_sync_before_inference)
    for (size_t i = 0; i < mgroup_input_tensors[grpid].size(); i++)
      axclrtMemcpy((void *)mgroup_input_tensors[grpid][i].phyAddr,
                   mgroup_input_tensors[grpid][i].pVirAddr,
                   mgroup_input_tensors[grpid][i].nSize,
                   AXCL_MEMCPY_HOST_TO_DEVICE);

  auto ret = axclrtEngineExecute(m_handle->handle, m_handle->context, grpid,
                                 m_handle->ios[grpid]);
  if (ret != 0) {
    fprintf(stderr, "axclrtEngineExecute failed. ret=0x%x\n", ret);
    return ret;
  }

  if (_auto_sync_after_inference)
    for (size_t i = 0; i < mgroup_output_tensors[grpid].size(); i++)
      axclrtMemcpy(mgroup_output_tensors[grpid][i].pVirAddr,
                   (void *)mgroup_output_tensors[grpid][i].phyAddr,
                   mgroup_output_tensors[grpid][i].nSize,
                   AXCL_MEMCPY_DEVICE_TO_HOST);
  return 0;
}
