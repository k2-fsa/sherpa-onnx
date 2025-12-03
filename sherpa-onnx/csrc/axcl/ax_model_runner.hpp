// sherpa-onnx/csrc/axcl/ax_model_runner.hpp
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#pragma once
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

typedef enum _color_space_e {
  ax_color_space_unknown,
  ax_color_space_nv12,
  ax_color_space_nv21,
  ax_color_space_bgr,
  ax_color_space_rgb,
} ax_color_space_e;

typedef struct {
  std::string sName;
  unsigned int nIdx;
  std::vector<unsigned int> vShape;
  int nSize;
  unsigned long long phyAddr;
  void *pVirAddr;
} ax_runner_tensor_t;

class ax_runner_base {
 public:
  std::vector<ax_runner_tensor_t> moutput_tensors;
  std::vector<ax_runner_tensor_t> minput_tensors;

  std::vector<std::vector<ax_runner_tensor_t>> mgroup_output_tensors;
  std::vector<std::vector<ax_runner_tensor_t>> mgroup_input_tensors;

  std::map<std::string, ax_runner_tensor_t> map_output_tensors;
  std::map<std::string, ax_runner_tensor_t> map_input_tensors;

  std::map<std::string, std::vector<ax_runner_tensor_t>>
      map_group_output_tensors;
  std::map<std::string, std::vector<ax_runner_tensor_t>>
      map_group_input_tensors;

  bool _auto_sync_before_inference = true;
  bool _auto_sync_after_inference = true;

  float cost_host_to_device = 0;
  float cost_inference = 0;
  float cost_device_to_host = 0;

 public:
  virtual int init(const char *model_file) = 0;
  virtual int init(char *model_buffer, size_t model_size) = 0;

  virtual void deinit() = 0;

  float get_inference_time() { return cost_inference; }

  int get_num_inputs() { return minput_tensors.size(); };
  int get_num_outputs() { return moutput_tensors.size(); };

  const ax_runner_tensor_t &get_input(int idx) { return minput_tensors[idx]; }
  const ax_runner_tensor_t *get_inputs_ptr() { return minput_tensors.data(); }
  const ax_runner_tensor_t &get_input(std::string name) {
    if (map_input_tensors.size() == 0) {
      for (size_t i = 0; i < minput_tensors.size(); i++) {
        map_input_tensors[minput_tensors[i].sName] = minput_tensors[i];
      }
    }
    if (map_input_tensors.find(name) == map_input_tensors.end()) {
      throw std::runtime_error("input tensor not found: " + name);
    }

    return map_input_tensors[name];
  }

  const ax_runner_tensor_t &get_input(int grpid, int idx) {
    return mgroup_input_tensors[grpid][idx];
  }
  const ax_runner_tensor_t *get_inputs_ptr(int grpid) {
    return mgroup_input_tensors[grpid].data();
  }
  const ax_runner_tensor_t &get_input(int grpid, std::string name) {
    if (map_group_input_tensors.size() == 0) {
      for (size_t i = 0; i < mgroup_input_tensors.size(); i++) {
        for (size_t j = 0; j < mgroup_input_tensors[i].size(); j++) {
          map_group_input_tensors[mgroup_input_tensors[i][j].sName].push_back(
              mgroup_input_tensors[i][j]);
        }
      }
    }
    if (map_group_input_tensors.find(name) == map_group_input_tensors.end()) {
      throw std::runtime_error("input tensor not found: " + name);
    }
    return map_group_input_tensors[name][grpid];
    // return map_input_tensors[name];
  }

  const ax_runner_tensor_t &get_output(int idx) { return moutput_tensors[idx]; }
  const ax_runner_tensor_t *get_outputs_ptr() { return moutput_tensors.data(); }
  const ax_runner_tensor_t &get_output(std::string name) {
    if (map_output_tensors.size() == 0) {
      for (size_t i = 0; i < moutput_tensors.size(); i++) {
        map_output_tensors[moutput_tensors[i].sName] = moutput_tensors[i];
      }
    }
    if (map_output_tensors.find(name) == map_output_tensors.end()) {
      throw std::runtime_error("output tensor not found: " + name);
    }

    return map_output_tensors[name];
  }

  const ax_runner_tensor_t &get_output(int grpid, int idx) {
    return mgroup_output_tensors[grpid][idx];
  }
  const ax_runner_tensor_t *get_outputs_ptr(int grpid) {
    return mgroup_output_tensors[grpid].data();
  }
  const ax_runner_tensor_t &get_output(int grpid, std::string name) {
    if (map_group_output_tensors.size() == 0) {
      for (size_t i = 0; i < mgroup_output_tensors.size(); i++) {
        for (size_t j = 0; j < mgroup_output_tensors[i].size(); j++) {
          map_group_output_tensors[mgroup_output_tensors[i][j].sName].push_back(
              mgroup_output_tensors[i][j]);
        }
      }
    }
    if (map_group_output_tensors.find(name) == map_group_output_tensors.end()) {
      throw std::runtime_error("input tensor not found: " + name);
    }
    return map_group_output_tensors[name][grpid];
  }

  virtual int get_algo_width() = 0;
  virtual int get_algo_height() = 0;
  virtual ax_color_space_e get_color_space() = 0;

  void set_auto_sync_before_inference(bool sync) {
    _auto_sync_before_inference = sync;
  }
  void set_auto_sync_after_inference(bool sync) {
    _auto_sync_after_inference = sync;
  }

  virtual int inference() = 0;
  virtual int inference(int grpid) = 0;
};
