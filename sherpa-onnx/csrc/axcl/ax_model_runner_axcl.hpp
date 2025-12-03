// sherpa-onnx/csrc/axcl/ax_model_runner_axcl.hpp
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#pragma once
#include "ax_model_runner.hpp"

class ax_runner_axcl : public ax_runner_base {
 protected:
  struct ax_joint_runner_axcl_handle_t *m_handle = nullptr;
  int group_count = 0;
  bool _parepare_io = false;

  int sub_init();

 public:
  int init(const char *model_file) override;
  int init(char *model_buffer, size_t model_size) override;

  void release();
  void deinit() override;

  int get_algo_width() override;
  int get_algo_height() override;
  ax_color_space_e get_color_space() override;

  int set_input(int grpid, int idx, unsigned long long int phy_addr,
                unsigned long size);
  int set_output(int grpid, int idx, unsigned long long int phy_addr,
                 unsigned long size);

  int set_input(int grpid, std::string name, unsigned long long int phy_addr,
                unsigned long size);
  int set_output(int grpid, std::string name, unsigned long long int phy_addr,
                 unsigned long size);

  // int inference(ax_image_t *pstFrame) override;
  int inference() override;
  int inference(int grpid) override;
};